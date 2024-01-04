import click
import numpy
import pandas

import math
import pickle

# In the run/jump data, one row is 151x3, corresponding to a label
# This will require a custom loader class also...
# Data might get too big for memory if we're talking 7500 time steps and more than 4 or so firing positions (4 x 7500)

# Output shape should be (batch, channel, 1, length)

# Note 1-based indexing for shell-types with 0 reserved for "no shell"

@click.command()
@click.option("--num-samples", default=1000, help="Number of sample shows to generate.")
@click.option("--num-shell-types", default=5, help="Number of unique shell types")
@click.option(
    "--time-step", default=0.25, help="Time series sample time in seconds(period)"
)
@click.option("--max-duration", default=90, help="Maximum show duration in seconds")
@click.option("--min-duration", default=70, help="Minimum show duration in seconds")
def make_data(num_samples, num_shell_types, time_step, max_duration, min_duration):
    num_channels = 3
    num_time_steps = math.ceil(max_duration / time_step)
    samples = numpy.zeros((num_samples, num_channels, 1, num_time_steps))
    time = numpy.array([j * time_step for j in range(num_time_steps)])

    for i in range(samples.shape[0]):
        samples[i][0] = make_saw_tooth(
            time, num_shell_types, min_duration, max_duration
        )
        samples[i][1] = make_sine(time, num_shell_types, min_duration, max_duration)
        samples[i][2] = make_constant(time, num_shell_types, min_duration, max_duration)

    data = {
        "samples": samples,
        "time": time,
        "time_step": time_step,
        "num_shell_types": num_shell_types,
        "max_duration": max_duration,
        "min_duration": min_duration,
    }

    with open("test_data.pkl", "wb") as f:
        pickle.dump(data, f)


def make_saw_tooth(time, num_shell_types, min_duration, max_duration):
    stride = numpy.random.randint(1, 8)
    idx = range(0, len(time), stride)
    show = numpy.zeros(time.shape)
    show[idx] = numpy.mod(range(0, len(idx)), num_shell_types) + 1
    print(show[0:25])
    return clip_show_time(show, time, min_duration, max_duration)


def make_sine(time, num_shell_types, min_duration, max_duration):
    show = numpy.round(
        num_shell_types * (numpy.sin(time * 2 * math.pi * 1 / 20) * 0.5 + 0.5) + 1
    )
    return clip_show_time(show, time, min_duration, max_duration)


def make_constant(time, num_shell_types, min_duration, max_duration):
    shell = numpy.random.randint(0, num_shell_types) + 1
    show = shell * numpy.ones(time.shape)
    return clip_show_time(show, time, min_duration, max_duration)


def clip_show_time(show, time, min_duration, max_duration):
    stop_time = min_duration + numpy.random.rand() * (max_duration - min_duration)
    show[time > stop_time] = 0
    return show


if __name__ == "__main__":
    make_data()
