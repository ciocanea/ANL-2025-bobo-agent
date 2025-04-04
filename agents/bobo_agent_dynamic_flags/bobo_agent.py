import logging
from random import randint
from time import time
from typing import cast

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


class BoboAgentDynamic(DefaultParty):
    """Adaptive negotiation agent with opponent modeling and dynamic strategies."""

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

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.opponent_utils = []

        # Weights replacing boolean flags
        self.greedy_weight = 0.0  # Opponent greediness: 0 (not greedy) to 1 (very greedy)
        self.nice_weight = 0.0  # Opponent niceness: 0 (not nice) to 1 (very nice)

        self.logger.log(logging.INFO, "Agent initialized")

    def notifyChange(self, data: Inform):
        """Handles negotiation events."""
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()
            self.progress = self.settings.getProgress()
            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            profile_connection.close()

        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()
            if actor != self.me:
                self.other = str(actor).rsplit("_", 1)[0]
                self.opponent_action(action)

        elif isinstance(data, YourTurn):
            self.my_turn()

        elif isinstance(data, Finished):
            self.save_data()
            self.logger.log(logging.INFO, "Negotiation ended")
            super().terminate()

        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """Returns agent capabilities."""
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent."""
        self.getConnection().send(action)

    def getDescription(self) -> str:
        """Returns a brief description of the agent."""
        return "Adaptive negotiation agent using opponent modeling and dynamic strategies."

    def opponent_action(self, action):
        """Processes an opponent's action."""
        if isinstance(action, Offer):
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()
            self.opponent_model.update(bid)
            self.last_received_bid = bid

            util = self.opponent_model.get_predicted_utility(bid)
            self.opponent_utils.append(util)

            if len(self.opponent_utils) >= 3:
                recent = self.opponent_utils[-5:]
                avg_util = sum(recent) / len(recent)
                progress = self.progress.get(time() * 1000)

                # Greedy weight (slower increase)
                self.greedy_weight = min(1.0, max(0.0, abs(avg_util - 0.5) * 2 - 0.15 * progress))

                # Nice weight (more stable, increases later)
                self.nice_weight = min(1.0, max(0.0, 1.3 - avg_util - 0.15 * progress))

    def my_turn(self):
        """Executes the next action."""
        if self.accept_condition(self.last_received_bid):
            action = Accept(self.me, self.last_received_bid)
        else:
            bid = self.find_bid()
            action = Offer(self.me, bid)

        self.send_action(action)

    def save_data(self):
        """Stores learning data."""
        data = "Learning data (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

    def accept_condition(self, bid: Bid) -> bool:
        """Determines whether to accept an offer."""
        if bid is None:
            return False

        progress = self.progress.get(time() * 1000)
        offered_util = self.profile.getUtility(bid)

        if offered_util >= 0.85:
            return True

        if self.greedy_weight < 0.5 and progress > 0.9 and offered_util > 0.7:
            return True

        if self.opponent_model:
            opponent_util = float(self.opponent_model.get_predicted_utility(bid))

            if progress > 0.95 and offered_util > 0.75 and opponent_util > 0.6:
                return True

            if progress > 0.99 and offered_util > 0.7 and opponent_util < 0.5:
                return True

        return offered_util > (0.75 if progress > 0.98 else 0.8 if progress > 0.95 else 0.85)

    def find_bid(self) -> Bid:
        """Finds a suitable bid."""
        domain = self.profile.getDomain()
        all_bids = AllBidsList(domain)

        best_bid_score = 0.0
        best_bid = None
        progress = self.progress.get(time() * 1000)
        min_util = 0.85 * (1 - 0.15 * progress)

        for _ in range(500):
            bid = all_bids.get(randint(0, all_bids.size() - 1))
            if self.profile.getUtility(bid) < min_util:
                continue

            bid_score = self.score_bid(bid)
            if bid_score > best_bid_score:
                best_bid_score, best_bid = bid_score, bid

        return best_bid or all_bids.get(randint(0, all_bids.size() - 1))

    def score_bid(self, bid: Bid) -> float:
        """Scores a bid based on opponent modeling and time pressure."""
        progress = self.progress.get(time() * 1000)
        our_utility = float(self.profile.getUtility(bid))
        opponent_utility = self.opponent_model.get_predicted_utility(bid) if self.opponent_model else 0

        alpha = max(0.6, 0.95 - 0.3 * progress * (1 - self.nice_weight))
        return alpha * our_utility + (1.0 - alpha) * opponent_utility * (1 - self.greedy_weight)
