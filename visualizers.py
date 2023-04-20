import matplotlib.pyplot as plt
import numpngw


class GIFVisualizer(object):
    def __init__(self):
        self.frames = []

    def set_data(self, img):
        self.frames.append(img)

    def reset(self):
        self.frames = []

    def get_gif(self, dir, obs, filename='pushing_visualization.gif'):
        # generate the gif
        filename = 'pushing_visualization.gif'
        print("Creating animated gif, please wait about 10 seconds")
        numpngw.write_apng(filename, self.frames, delay=10)
        if obs == 0:
            log_name =  dir+"/pushing_visualization_free.gif"
        if obs == 1:
            log_name =  dir+"/pushing_visualization_obs.gif"
        numpngw.write_apng(log_name, self.frames, delay=10)
        return filename


class NotebookVisualizer(object):
    def __init__(self, fig, hfig):
        self.fig = fig
        self.hfig = hfig

    def set_data(self, img):
        plt.clf()
        plt.imshow(img)
        plt.axis('off')
        self.fig.canvas.draw()
        self.hfig.update(self.fig)

    def reset(self):
        pass