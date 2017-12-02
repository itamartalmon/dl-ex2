from matplotlib import pyplot


def plot_learning_curves(train_losses, test_losses, name):

    fig = pyplot.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_prop_cycle('color', ['c', 'm'])

    ax.plot(train_losses, label='{} ({})'.format(name, 'train'), linewidth=3)
    ax.plot(test_losses, label='{} ({})'.format(name, 'test'), linestyle='--', linewidth=3)


    ax.set_xticks([])

    ax.legend()
    ax.grid()
    ax.set_ylim(bottom=0.005, top=0.01)
    pyplot.yscale('log')
    pyplot.ylabel('CrossEntropy')
    pyplot.xlabel('Epochs')
    pyplot.show()
    fig.savefig('{}.png'.format(name))
