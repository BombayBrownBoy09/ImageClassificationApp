from training import classified_image
from click.testing import CliRunner


def test_classified_image():
    result = classified_image(
        "http://images.cocodataset.org/val2017/000000039769.jpg"
    )
    assert result.lower.__contains__('cat')
  
