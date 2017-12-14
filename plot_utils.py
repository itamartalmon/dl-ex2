from matplotlib import pyplot


def plot_learning_curves(train_losses, test_losses, name):

    fig = pyplot.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_prop_cycle('color', ['c', 'm'])

    ax.plot(train_losses, label='{} ({})'.format(name, 'train'), linewidth=3)
    ax.plot(test_losses, label='{} ({})'.format(name, 'test'), linestyle='--', linewidth=3)


    # ax.set_xticks([])

    ax.legend()
    ax.grid()
    # ax.set_ylim(bottom=0.002, top=0.005)
    pyplot.yscale('log')
    pyplot.ylabel('CrossEntropy Loss')
    pyplot.xlabel('Epochs')
    fig.savefig('{}.png'.format(name))


def box2ellipsePIL(box, height, width):
    xmin = box[0]
    ymin = box[1] * 0.8
    xmax = min(box[2], width)
    ymax = min(box[3] * 1.2, height)
    return xmin, ymin, xmax, ymax


def box2ellipse(box, height, width):
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    major_axis_radius = (xmax - xmin) * 0.5
    minor_axis_radius = (ymax - ymin) * 0.5 * 1.2
    angle = 0.0
    center_x = (xmin + xmax) * 0.5
    center_y = (ymin + ymax) * 0.5 - minor_axis_radius * 0.2
    detection_score = box[4]
    return '{} {} {} {} {} {}'.format(major_axis_radius, minor_axis_radius, angle, center_x, center_y, detection_score)
