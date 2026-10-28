import time


class ReplayEngine:

    def __init__(
        self,
        dataframe,
        speed=100
    ):
        self.df = dataframe
        self.speed = speed

        self.current_position = 0

    def stream(self):

        for timestamp, row in self.df.iterrows():

            yield timestamp, row

            time.sleep(
                2 / self.speed
            )