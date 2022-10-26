import numpy as np


def create_input_order(input_size, input_order="left-to-right"):
    """Returns a degree vectors for the input."""
    if input_order == "left-to-right":
        return np.arange(start=1, stop=input_size + 1)
    elif input_order == "right-to-left":
        return np.arange(start=input_size, stop=0, step=-1)
    elif input_order == "random":
        ret = np.arange(start=1, stop=input_size + 1)
        np.random.shuffle(ret)
        return ret


def create_degrees(
    input_size, hidden_units, input_order="left-to-right", hidden_degrees="equal"
):
    input_order = create_input_order(input_size, input_order)
    degrees = [input_order]
    for units in hidden_units:
        if hidden_degrees == "random":
            # samples from: [low, high)
            degrees.append(
                np.random.randint(
                    low=min(np.min(degrees[-1]), input_size - 1),
                    high=input_size,
                    size=units,
                )
            )
        elif hidden_degrees == "equal":
            min_degree = min(np.min(degrees[-1]), input_size - 1)
            degrees.append(
                np.maximum(
                    min_degree,
                    # Evenly divide the range `[1, input_size - 1]` in to `units + 1`
                    # segments, and pick the boundaries between the segments as degrees.
                    np.ceil(
                        np.arange(1, units + 1) * (input_size - 1) / float(units + 1)
                    ).astype(np.int32),
                )
            )
    return degrees


def create_masks(degrees):
    """Returns a list of binary mask matrices enforcing autoregressivity."""
    return [
        # Create input->hidden and hidden->hidden masks.
        inp[:, np.newaxis] <= out
        for inp, out in zip(degrees[:-1], degrees[1:])
    ] + [
        # Create hidden->output mask.
        degrees[-1][:, np.newaxis]
        < degrees[0]
    ]
