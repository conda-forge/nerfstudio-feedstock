import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import components_from_spherical_harmonics


def test_colors():
    torch.testing.assert_close(get_color("red"), torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(get_color([0.25, 0.5, 0.75]), torch.tensor([0.25, 0.5, 0.75]))


def test_scene_box():
    box = SceneBox(aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))

    torch.testing.assert_close(box.get_center(), torch.zeros(3))
    torch.testing.assert_close(box.get_diagonal_length(), torch.sqrt(torch.tensor(12.0)))
    assert box.within(torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])).tolist() == [True, False]

    normalized = SceneBox.get_normalized_positions(torch.tensor([[0.0, 0.0, 0.0]]), box.aabb)
    torch.testing.assert_close(normalized, torch.tensor([[0.5, 0.5, 0.5]]))


def test_ray_samples():
    rays = RayBundle(
        origins=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        directions=torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        pixel_area=torch.ones((2, 1)),
    )
    rays.set_camera_indices(3)

    starts = torch.tensor([[[0.0], [1.0]], [[0.0], [2.0]]])
    ends = torch.tensor([[[1.0], [2.0]], [[1.0], [3.0]]])
    samples = rays.get_ray_samples(starts, ends)

    assert len(rays) == 2
    assert samples.frustums.get_positions().shape == (2, 2, 3)
    torch.testing.assert_close(samples.frustums.get_positions()[0, :, 2], torch.tensor([0.5, 1.5]))
    assert samples.camera_indices.squeeze().tolist() == [[3, 3], [3, 3]]

    weights = samples.get_weights(torch.ones((2, 2, 1)))
    assert weights.shape == (2, 2, 1)
    assert bool(torch.all(weights >= 0))


def test_spherical_harmonics_and_embedding():
    components = components_from_spherical_harmonics(2, torch.tensor([[0.0, 0.0, 1.0]]))
    assert components.shape == (1, 4)
    torch.testing.assert_close(components[0, :2], torch.tensor([0.28209478, 0.0]))
    torch.testing.assert_close(components[0, 2], torch.tensor(0.48860252))

    embedding = Embedding(in_dim=3, out_dim=2)
    assert embedding(torch.tensor([0, 1])).shape == (2, 2)
    assert embedding.mean().shape == (2,)


test_colors()
test_scene_box()
test_ray_samples()
test_spherical_harmonics_and_embedding()
