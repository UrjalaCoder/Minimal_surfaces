from manim import *
from ..grid import Grid
from ..vector_field import VectorField

def map_to_camera_view(point: np.array) -> np.array:
    return (point * 10) - 5

def inverse_map_to_camera_view(point: np.array) -> np.array:
    return (point + 5) / 10

def build_vector_field_scene(grid: Grid, vector_field, curve_points,
                              vector_scale=0.1, color=BLUE, max_vectors=1000, streak_time = 0.8, streak_opacity = 0.6):
    
    def mapping_of_vector_field(camera_position: np.array) -> np.array:
        mapped_position = inverse_map_to_camera_view(camera_position)
        return vector_field(mapped_position)
    
    def mapping_of_curve(curve_points: np.array) -> np.array:
        return np.array([map_to_camera_view(p) for p in curve_points])

    class DiracStreamLineScene(ThreeDScene):
        def construct(self):
            self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
            self.camera.set_zoom(1.0)

            # Add the curve
            mapped_curve_points = mapping_of_curve(curve_points)
            curve = VMobject(color=WHITE).set_points_smoothly(mapped_curve_points)
            self.add(curve)

            # Add streamlines
            stream_lines = StreamLines(
                func=mapping_of_vector_field,
                x_range=[-5, 5, 1],
                y_range=[-5, 5, 1],
                z_range=[-4, 4, 1],
                stroke_width=1.2,
                virtual_time=streak_time,
                color=color,
                opacity=streak_opacity,
                dt = 0.05
            )
            stream_lines.start_animation()
            self.add(stream_lines)
            self.wait(10)
 
    return DiracStreamLineScene
