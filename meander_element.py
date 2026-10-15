import numpy as np
from manim import *  # pyright: ignore[reportWildcardImportFromLibrary]
from seaweed_algebra import Meander

from meander_arc import MeanderArc, MeanderNode


class MeanderElement(VGroup):
    def __init__(self, meander: Meander, **kwargs):
        super().__init__(**kwargs)

        self.meander = meander
        total_nodes = sum(self.meander.top_blocks)

        # 1. Create single source-of-truth nodes (0-indexed internally)
        self.nodes: list[MeanderNode] = [MeanderNode(i) for i in range(total_nodes)]

        # 2. Create top and bottom arcs linked to self.nodes
        self.top_arcs = self._gen_arcs(self.meander.top_blocks, is_top=True)
        self.btm_arcs = self._gen_arcs(self.meander.bottom_blocks, is_top=False)

        # Add arcs first, then nodes on top so dots render cleanly over arc tips
        self.add(*self.btm_arcs, *self.top_arcs, *self.nodes)

    def _gen_arcs(self, blocks: list[int], is_top: bool = True) -> list[MeanderArc]:
        start_offset = 0
        arcs = []

        for block in blocks:
            size = block - 1
            arc_offset = 0

            while size >= 1:
                left = start_offset + arc_offset
                right = start_offset + arc_offset + size

                arcs.append(
                    MeanderArc(
                        left_node=self.nodes[left],
                        right_node=self.nodes[right],
                        is_top=is_top,
                        arc_color=BLUE,
                    )
                )

                size -= 2
                arc_offset += 1
            start_offset += block

        return arcs

    def _retire_mobject(self, mob: Mobject):
        """
        Safely removes a mobject from this VGroup AND hides it completely in case
        Manim's Scene promoted it to a top-level mobject during self.play().
        """
        mob.clear_updaters()
        mob.set_opacity(0)
        mob.clear_points()
        self.remove(mob)

    def cleanup_after_collapse(
        self,
        remove_arcless_nodes: bool = False,
        tol: float = 1e-2,
    ):
        """
        Call right after self.play(...) of a collapse/rotation move.
        1. Retires all collapsed arcs (both flagged and coincident endpoints).
        2. Merges overlapping nodes and re-links surviving arcs.
        3. Retires any arcs that collapsed due to node merging.
        4. Resets node tangents to vertical (PI/2) and normalizes flipped arcs.
        5. Optionally removes any node that no longer has any arcs attached.
        6. Sorts surviving nodes left-to-right and re-indexes 0..N-1.
        """

        # 1. Retire collapsed arcs (flagged or endpoints already coincident)
        all_arcs = self.top_arcs + self.btm_arcs
        surviving_arcs: list[MeanderArc] = []
        for arc in all_arcs:
            dist = float(
                np.linalg.norm(arc.left_node.get_center() - arc.right_node.get_center())
            )
            if arc.is_collapsing or dist < tol:
                self._retire_mobject(arc)
            else:
                surviving_arcs.append(arc)

        # 2. Merge overlapping nodes
        surviving_nodes: list[MeanderNode] = []
        for node in self.nodes:
            merged_target = None
            for target in surviving_nodes:
                if np.linalg.norm(node.get_center() - target.get_center()) < tol:
                    merged_target = target
                    break

            if merged_target is not None:
                for arc in surviving_arcs:
                    arc.replace_node(node, merged_target)
                self._retire_mobject(node)
            else:
                surviving_nodes.append(node)

        # 3. Retire any arcs that collapsed due to node merging
        active_arcs: list[MeanderArc] = []
        for arc in surviving_arcs:
            dist = float(
                np.linalg.norm(arc.left_node.get_center() - arc.right_node.get_center())
            )
            if arc.left_node is arc.right_node or dist < tol:
                self._retire_mobject(arc)
            else:
                active_arcs.append(arc)
        surviving_arcs = active_arcs

        # 4. Reset node tangents to vertical (PI/2) and clear old arc pointers
        for node in surviving_nodes:
            node.reset_tangent()
            node.top_arc = None
            node.btm_arc = None

        # 5. Normalize flipped arcs and re-register with their nodes
        self.top_arcs = []
        self.btm_arcs = []
        for arc in surviving_arcs:
            arc.normalize_orientation()
            arc.left_node.set_arc(arc.is_top, arc)
            arc.right_node.set_arc(arc.is_top, arc)
            if arc.is_top:
                self.top_arcs.append(arc)
            else:
                self.btm_arcs.append(arc)

        # 6. Optionally prune nodes that have neither a top nor a bottom arc
        if remove_arcless_nodes:
            active_nodes = []
            for node in surviving_nodes:
                if node.top_arc is None and node.btm_arc is None:
                    self._retire_mobject(node)
                else:
                    active_nodes.append(node)
            surviving_nodes = active_nodes

        # 7. Sort surviving nodes left-to-right and re-index 0..N-1
        surviving_nodes.sort(key=lambda n: n.get_center()[0])
        for new_idx, node in enumerate(surviving_nodes):
            node.idx = new_idx
        self.nodes = surviving_nodes

        # Force one immediate update on surviving arcs so their Bezier handles
        # reflect the normalized orientation and reset tangents seamlessly
        for arc in surviving_arcs:
            arc.update()

    def slide_to_compact(self, start_x: float = 0.0, **kwargs) -> AnimationGroup:
        """
        Slides all surviving nodes horizontally to contiguous integer coordinates
        (start_x, start_x + 1, ...), eliminating any gaps.
        """
        anims = []
        for i, node in enumerate(self.nodes):
            target_x = start_x + i
            if abs(node.get_center()[0] - target_x) > 1e-4:
                anims.append(node.animate.move_to((target_x, 0, 0)))
        return AnimationGroup(*anims, **kwargs)

    def rotation_contraction(self, **kwargs) -> AnimationGroup:

        a1 = self.meander.top_blocks[0]
        b1 = self.meander.bottom_blocks[0]
        num_collapsing = a1 - b1

        return AnimationGroup(
            *(
                node.move_along_arc(is_top=True, collapse_arc=False)
                for node in self.nodes[:num_collapsing]
            ),
            *(
                node.move_along_arc(is_top=False, collapse_arc=False)
                for node in self.nodes[num_collapsing:b1]
            ),
            **kwargs,
        )

    def fit_camera(self, scene: Scene) -> None:
        scene.camera.frame.move_to(self.get_center())  # type: ignore
        scene.camera.frame.width = (
            max(*self.meander.top_blocks, *self.meander.bottom_blocks) * 2
        )  # type: ignore

    def fit_camera_animation(self, scene: Scene) -> Animation:
        return scene.camera.frame.animate.move_to(self.get_center()).set(  # type: ignore
            width=max(*self.meander.top_blocks, *self.meander.bottom_blocks) * 2
        )

    def rotation_contraction_animation(self, scene: Scene, **kwargs) -> None:

        scene.play(Succession(self.rotation_contraction(run_time=2.0), Wait(0.5)))

        self.cleanup_after_collapse()

        scene.play(
            self.slide_to_compact(run_time=1.5),
        )

        self.meander = self.meander._rotation_contraction()

        scene.play(
            self.fit_camera_animation(scene),
        )

