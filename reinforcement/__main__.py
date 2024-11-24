from datetime import datetime
import logging

import numpy as np

from reinforcement.agent_QLearning import AgentQ
from reinforcement.agent_Sarsa import AgentSarsa
from reinforcement.joc import Laberint

import sys
sys.path.append("/Documents/3r & 4t/Intel·ligència Artificial/ia_2024")

def main():
    """
    logging.basicConfig(
        format="%(levelname)-8s: %(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )  # Only show messages *equal to or above* this level
    """

    game = Laberint()
    eps = []

    for n in range(25):
        #agent = AgentQ(game)
        agent = AgentSarsa(game)
        start = datetime.now()

        h, w, _, d = agent.train(
            discount=0.2,
            exploration_rate=0.10,
            learning_rate=0.6,
            episodes=1000,
            stop_at_convergence=False,
        )

        #agent.print_Q()
        print(d - start)

        """
        for i in range(0, 250):
            print(f"({i}, {h[i]:.2f})", end=" ")
        print()
        """

        i = 0
        while h[i] > h[i+1]:
            i += 1

        eps.append(i+1)
        #print(i)

    print(eps)
    print(f"Media de episodios: {np.mean(eps)}")

    """
    game.reset(start_cell=(0, 0))
    game.set_agent([agent])
    game.comencar()
    """


if __name__ == "__main__":
    main()
