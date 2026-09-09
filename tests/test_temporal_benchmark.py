from minimind_lab.data.temporal_benchmark import _changing_size, _event_order, _moving_shape


class FixedChoice:
    def choice(self, values):
        return values[0]


def center_of_non_background(image) -> tuple[float, float]:
    pixels = image.load()
    background = pixels[0, 0]
    locations = [(x, y) for y in range(image.height) for x in range(image.width) if pixels[x, y] != background]
    return (
        sum(location[0] for location in locations) / len(locations),
        sum(location[1] for location in locations) / len(locations),
    )


def foreground_area(image) -> int:
    pixels = image.load()
    background = pixels[0, 0]
    return sum(pixels[x, y] != background for y in range(image.height) for x in range(image.width))


def test_motion_frames_encode_requested_direction():
    frames = _moving_shape(FixedChoice(), "right")
    assert center_of_non_background(frames[0])[0] < center_of_non_background(frames[-1])[0]
    frames = _moving_shape(FixedChoice(), "up")
    assert center_of_non_background(frames[0])[1] > center_of_non_background(frames[-1])[1]


def test_size_frames_encode_requested_change():
    larger = _changing_size(FixedChoice(), "larger")
    smaller = _changing_size(FixedChoice(), "smaller")
    assert len(larger) == len(smaller) == 8
    assert foreground_area(larger[0]) < foreground_area(larger[-1])
    assert foreground_area(smaller[0]) > foreground_area(smaller[-1])


def test_event_order_frames_encode_requested_first_event():
    red_first = _event_order(FixedChoice(), "red square")
    blue_first = _event_order(FixedChoice(), "blue circle")
    assert red_first[0].getpixel((128, 128))[0] > red_first[0].getpixel((128, 128))[2]
    assert blue_first[0].getpixel((128, 128))[2] > blue_first[0].getpixel((128, 128))[0]
