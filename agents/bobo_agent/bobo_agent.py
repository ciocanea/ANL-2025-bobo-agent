import logging
from random import randint
from time import time
from typing import cast
import os

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.opponent_model import OpponentModel


class BoboAgent(DefaultParty):
    """
    Template of a Python geniusweb agent.
    """

    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None
        self.opponent_utils = []  # Store all predicted utilities of opponent's offers
        self.opponent_is_greedy = False  # Flag is the robot is a bully (utility always over 0.9)
        self.opponent_is_nice = False  # Flag for nice robots

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.logger.log(logging.INFO, "party is initialized")

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        # if self.opponent_model and self.progress:
        #     if self.opponent_model.get_concession_trend() < 0.01:
        #         self.opponent_is_greedy = True
        #     else:
        #         self.opponent_is_greedy = False

        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            self.load_opponent_data()
            profile_connection.close()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()


            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def load_opponent_data(self):
        """Loads opponent history from data.md if available."""
        data_path = os.path.join(self.storage_dir, "data.md")
        if not os.path.exists(data_path):
            return

        with open(data_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 3 and parts[0] == self.other:
                    self.opponent_is_greedy = parts[1] == "True"
                    self.opponent_is_nice = parts[2] == "True"
                    break  # Stop searching once found
    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Template agent for the ANL 2022 competition"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()
            self.opponent_model.update(bid)
            self.last_received_bid = bid

            # track opponent utility for their own offer
            util = self.opponent_model.get_predicted_utility(bid)
            self.opponent_utils.append(util)

            # Check if they are greedy (after at least 3 offers)
            if len(self.opponent_utils) >= 3:
                recent = self.opponent_utils[-5:]
                avg_util = sum(recent) / len(recent)
                if avg_util > 0.87:  # average utils so we consider a robot bully and change our appraoch
                    self.opponent_is_greedy = True
                if 0.6 < avg_util < 0.9:
                    self.opponent_is_nice = True

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self.find_bid()
            action = Offer(self.me, bid)

        # send the action
        self.send_action(action)

    def save_data(self):
        """Stores opponent behavior data in data.md."""
        if self.other is None:
            return  # No opponent data to save

        data_path = os.path.join(self.storage_dir, "data.md")
        opponent_data = f"{self.other},{self.opponent_is_greedy},{self.opponent_is_nice}\n"

        # Append data without overwriting
        with open(data_path, "a") as f:
            f.write(opponent_data)

    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################

    # bad approach, selfish acceptence crieterias, only when our utility is over 0.8 and progress is high, we could have a winning bid earlier in the game, or we could cooperate with our oponent to get a high utility score together
    # def accept_condition(self, bid: Bid) -> bool:
    #     if bid is None:
    #         return False

    #     # progress of the negotiation session between 0 and 1 (1 is deadline)
    #     progress = self.progress.get(time() * 1000)

    #     # very basic approach that accepts if the offer is valued above 0.7 and
    #     # 95% of the time towards the deadline has passed
    #     conditions = [
    #         self.profile.getUtility(bid) > 0.8,
    #         progress > 0.95,
    #     ]
    #     return all(conditions)

    def accept_condition(self, bid: Bid) -> bool:
        # if there's no bid, nothing to accept
        if bid is None:
            return False

        # calculate progress and the utility we are expecting
        progress = self.progress.get(time() * 1000)
        offered_util = self.profile.getUtility(bid)

        # if the offer is really good, we can accept straight away, no need to gamble for more
        if offered_util >= 0.9:
            return True

        if self.opponent_is_greedy and offered_util > 0.65:
            return True  # Take the decent deal while you can

        if self.opponent_model:
            opponent_util = float(self.opponent_model.get_predicted_utility(bid))

            # --- Handle hardliner bots ---
            # If opponent never concedes and offers bad utility, do NOT accept, even at the end
            if self.opponent_is_greedy:
                if progress > 0.99 and offered_util > 0.6:
                    return True  # last resort, avoid 0 if offer is not too bad
                elif progress > 0.97 and offered_util > 0.7:
                    return True
                else:
                    return False # we dont accept anything less

            # If both sides get 0.75+, accept — it’s fair
            if self.opponent_is_nice and progress > 0.95:
                if offered_util > 0.75 and opponent_util > 0.7:
                    return True

            # If opponent is conceding early and we got something decent
            if progress > 0.6 and offered_util > 0.6 and opponent_util < 0.5:
                return True

            # Accept near end if both sides are being fair
            if progress > 0.97 and float(offered_util) > 0.85 and float(opponent_util) < 0.6:
                return True  # they're conceding, let's lock the win

            # If both agents do well together, accept Nash-like reasoning
            if not self.opponent_is_greedy and float(offered_util) * float(opponent_util) > 0.75:
                return True

            # Reject greedy offers late in the game
            if progress > 0.95 and opponent_util > 0.9 and offered_util < 0.7:
                return False  # opponent is being greedy, don’t accept low

            # Accept if opponent finally concedes and we got something okay
            if progress > 0.93 and offered_util > 0.65 and opponent_util < 0.5:
                return True

        # worst case, if we are close to the deadline and nothing accepted so far, we accept a decent utility score to avoid negotiation failure, further from the
        # deadline, we need more utility to accept
        if progress > 0.98:  # was 0.99 before, this gives a bit more time to respond
            return offered_util > 0.70
        elif progress > 0.97:
            return offered_util > 0.75
        elif progress > 0.96:
            return offered_util > 0.8
        elif progress > 0.95:
            return offered_util > 0.85
        return False

    # this method finds a good bid to send the oponent, using heuristics and sampling
    def find_bid(self) -> Bid:
        # this gets the current negotiation domain
        domain = self.profile.getDomain()
        # all possible bids in the domain (its finite)
        all_bids = AllBidsList(domain)

        # initalize variables for later
        best_bid_score = 0.0
        best_bid = None

        # progress so far in the bid wars
        progress = self.progress.get(time() * 1000)
        # we define the minimum utility we would want a bid to have to be considered, it decreases at time passes since we are getting close to the
        min_util = 0.85 * (1 - 0.25 * progress)  # more assertive min util

        # we are random sampling 500 binds
        for _ in range(500):
            bid = all_bids.get(randint(0, all_bids.size() - 1))

            # ensure we don't concede too much unless we have to, if its too low we skip it based on the minimum utility we calculated for this round
            if self.profile.getUtility(bid) < min_util:
                continue

            # we score the bid based on the score bid method which combines our utilities together and time pressure into a score
            bid_score = self.score_bid(bid)

            # if this is the best bid we have seen so far, we store it
            if bid_score > best_bid_score:
                best_bid_score, best_bid = bid_score, bid

        # if no bid has been chosen, we pick a random one to avoid sending nothing
        if best_bid is None:
            best_bid = all_bids.get(randint(0, all_bids.size() - 1))  # safety

        return best_bid

    def score_bid(self, bid: Bid, alpha: float = 0.95, eps: float = 0.1) -> float:
        """Calculate heuristic score for a bid

        Args:
            bid (Bid): Bid to score
            alpha (float, optional): Trade-off factor between self interested and
                altruistic behaviour. Defaults to 0.95.
            eps (float, optional): Time pressure factor, balances between conceding
                and Boulware behaviour over time. Defaults to 0.1.

        Returns:
            float: score
        """
        if self.opponent_model is None:
            # fallback: only consider our own utility
            our_utility = float(self.profile.getUtility(bid))
            progress = self.progress.get(time() * 1000)
            time_pressure = 1.0 - progress ** (1 / eps)
            return alpha * time_pressure * our_utility

        progress = self.progress.get(time() * 1000)

        if self.opponent_model is not None:
            opponent_utility = self.opponent_model.get_predicted_utility(bid)

        # we penalize bids with very low predicted utility for the oponent, especially early in the session because oponent would be sure to reject (if its not a greedy robot)
        if not self.opponent_is_greedy:
            rejection_threshold = 0.1 - 0.08 * progress
            if opponent_utility < rejection_threshold:
                return 0.0
        else:
            opponent_utility = 0.5  # fallback

        # # Dynamically adjust alpha over time
        # if progress < 0.2:
        #     alpha = 0.95  # starting off strong
        # elif progress < 0.8:
        #     alpha = 0.75  # common ground between having a high utility score and cooperating with the oponent
        # else:
        #     alpha = 0.5   # fully cooperating with the oponent to find a common point to make a deal

        if self.opponent_is_nice:
            alpha = 0.7  # favor fairness more
        elif self.opponent_is_greedy:
            alpha = 0.95  # favor ourselves
        else:
            alpha = 0.95 - 0.45 * progress

        if self.opponent_is_greedy:
            # ignore their utility if is greedy,  just care about our utility
            return float(self.profile.getUtility(bid))

        # calculate our otility at the moment based on this bid
        our_utility = float(self.profile.getUtility(bid))

        # calculate if there's any time pressure or not
        time_pressure = 1.0 - progress ** (1 / eps)
        # the score will now be based on how far we are into the game, how much the agent cares about time in its decision making and our utility in this bid
        score = alpha * time_pressure * our_utility

        if self.opponent_model is not None:
            # get oponent's predicted utility
            opponent_utility = self.opponent_model.get_predicted_utility(bid)
            # calculate the oponents score to complement our score to get to a total of 1 potentially
            opponent_score = (1.0 - alpha * time_pressure) * opponent_utility
            score += opponent_score

        # return the expected score between both robots based on the moment of the game
        return score