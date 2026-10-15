import numpy as np
from manim import *  # pyright: ignore[reportWildcardImportFromLibrary]


class MeanderNode(Dot):
    def __init__(self, idx: int, **kwargs):
        super().__init__(point=(idx, 0, 0), **kwargs)
        self.idx = idx
        self.top_arc: MeanderArc | None = None
        self.btm_arc: MeanderArc | None = None

        # Angle of the "upward" tangent vector at this node (default: PI/2 = UP).
        # Top arcs exit along +T(tangent_angle); bottom arcs exit along -T(tangent_angle).
        self.tangent_angle: float = PI / 2

    def get_arc(self, is_top: bool) -> "MeanderArc | None":
        return self.top_arc if is_top else self.btm_arc

    def set_arc(self, is_top: bool, arc: "MeanderArc | None"):
        if is_top:
            self.top_arc = arc
        else:
            self.btm_arc = arc

    def get_tangent_vector(self, is_top: bool) -> np.ndarray:
        """
        Returns the unit control-handle direction vector for a top or bottom arc
        connected to this node.
        """
        vec = np.array([np.cos(self.tangent_angle), np.sin(self.tangent_angle), 0.0])
        return vec if is_top else -vec

    def reset_tangent(self):
        self.tangent_angle = PI / 2

    def move_along_arc(self, is_top: bool, collapse_arc: bool = False) -> Animation:
        """
        Moves this node along its connected top or bottom arc to the partner node's
        starting position, rotating its tangent vector smoothly by 180 degrees.
        If collapse_arc=True, the arc itself also shrinks to 0 behind the node.
        """
        arc = self.get_arc(is_top)
        if arc is None:
            raise ValueError(
                f"Node {self.idx} has no {'top' if is_top else 'bottom'} arc."
            )

        moving_left_to_right = self is arc.left_node
        path = arc.get_directed_path(left_to_right=moving_left_to_right)

        dir_sign = 1 if moving_left_to_right else -1
        top_sign = 1 if arc.is_top else -1
        start_theta = float(self.tangent_angle)
        delta_theta = -dir_sign * top_sign * PI

        def _move_and_rotate(mob: "MeanderNode", alpha: float):
            mob.move_to(path.point_from_proportion(alpha))
            mob.tangent_angle = start_theta + alpha * delta_theta

        node_anim = UpdateFromAlphaFunc(self, _move_and_rotate)
        if collapse_arc:
            return AnimationGroup(
                node_anim, arc.collapse(left_to_right=moving_left_to_right)
            )
        return node_anim


class MeanderArc(VMobject):
    def __init__(
        self,
        left_node: MeanderNode,
        right_node: MeanderNode,
        is_top: bool = True,
        arc_color: ManimColor = BLUE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.left_node = left_node
        self.right_node = right_node
        self.is_top = is_top
        self.arc_color = arc_color

        # Collapse state
        self.is_collapsing = False
        self.collapse_left_to_right = True
        self.collapse_alpha: float = 0.0
        self._frozen_curve: CubicBezier | None = None

        # Register this arc with its endpoint nodes
        self.left_node.set_arc(is_top, self)
        self.right_node.set_arc(is_top, self)

        def update_arc(mob: "MeanderArc"):
            if mob.is_collapsing and mob._frozen_curve is not None:
                alpha = float(np.clip(mob.collapse_alpha, 0.0, 1.0))
                if alpha >= 1.0 - 1e-4:
                    end_pt = (
                        mob._frozen_curve.get_end()
                        if mob.collapse_left_to_right
                        else mob._frozen_curve.get_start()
                    )
                    mob.become(
                        CubicBezier(end_pt, end_pt, end_pt, end_pt, color=mob.arc_color)
                    )
                elif mob.collapse_left_to_right:
                    mob.pointwise_become_partial(mob._frozen_curve, alpha, 1.0)
                else:
                    mob.pointwise_become_partial(mob._frozen_curve, 0.0, 1.0 - alpha)
            else:
                mob.become(mob._build_bezier())

        self.add_updater(update_arc)
        update_arc(self)

    def _build_bezier(self) -> CubicBezier:
        p_l = self.left_node.get_center()
        p_r = self.right_node.get_center()
        dist = float(np.linalg.norm(p_r - p_l))
        handle_len = 0.6 * dist

        t_l = self.left_node.get_tangent_vector(self.is_top)
        t_r = self.right_node.get_tangent_vector(self.is_top)

        h1 = p_l + handle_len * t_l
        h2 = p_r + handle_len * t_r

        return CubicBezier(p_l, h1, h2, p_r, color=self.arc_color)

    def get_directed_path(self, left_to_right: bool = True) -> CubicBezier:
        curve = self._build_bezier()
        if not left_to_right:
            curve.reverse_direction()
        return curve

    def collapse(self, left_to_right: bool = True) -> Animation:
        """
        Freezes the current Bezier curve and slices it from alpha -> 1.0 (or 0.0 -> 1-alpha)
        directly on `self` via UpdateFromAlphaFunc.
        """
        self._frozen_curve = self._build_bezier()
        self.is_collapsing = True
        self.collapse_left_to_right = left_to_right
        self.collapse_alpha = 0.0

        def _update_collapse(mob: "MeanderArc", alpha: float):
            mob.collapse_alpha = alpha

        return UpdateFromAlphaFunc(self, _update_collapse)

    def replace_node(self, old_node: MeanderNode, new_node: MeanderNode):
        if self.left_node is old_node:
            self.left_node = new_node
        elif self.right_node is old_node:
            self.right_node = new_node

    def normalize_orientation(self):
        if self.left_node.get_center()[0] > self.right_node.get_center()[0]:
            self.left_node, self.right_node = self.right_node, self.left_node
            self.is_top = not self.is_top
