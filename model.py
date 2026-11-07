# check if pyflamegpu is installed and if not install it
# import importlib.util
# if importlib.util.find_spec('pyflamegpu') is None:
#     import sys
#     !{sys.executable} -m pip install --extra-index-url https://whl.flamegpu.com/whl/cuda130/ pyflamegpu

# Import pyflamegpu and some other libraries we will use in the tutorial
import pyflamegpu
import pyflamegpu.codegen
import sys, random, math
import numpy as np
import matplotlib.pyplot as plt
import importlib
import itertools
import shutil
import os
from tqdm import tqdm

#set matplotlib backend to widget for some advanced plotting tools
# %matplotlib widget

## ------------------------
## Model parameters
## ------------------------
DIM = 50                # Dimensions of our 2D grid
SIMULATION_STEPS = 100  # Number of time steps
DENSITY = 0.8           # Density of houses
OCCUPANCY = 0.8         # Occupancy rate at initialisation
CAPITAL = [300000, 300001]
PRICE = [200000, 200001]
SEED = random.randint(1, 9999999999)


## ------------------------
## Global parameters
## ------------------------
INTEREST = 3.7
LTV = 90
CAPITAL_THRESHOLD = 200000

## --------------------------------
## Model, environment and agents
## --------------------------------
def create_model():
    model = pyflamegpu.ModelDescription("housing")
    return model

def define_environment(model):
    env = model.Environment()

    # Env properties
    env = model.Environment()
    env.newPropertyFloat("INTEREST", INTEREST)
    env.newPropertyInt("LTV", LTV)
    env.newPropertyInt("CAPITAL_THRESHOLD", CAPITAL_THRESHOLD)
    env.newPropertyInt("ID_COUNTER", 0)

def define_agents(model):
    """
    House agent
    """
    ## House
    h = model.newAgent("house")
    # print("define_agents")
    # print(type(h))
    h.newVariableInt("constructed?")
    h.newVariableInt("x")
    h.newVariableInt("y")
    h.newVariableInt("my_owner")    # ID of the household
    h.newVariableInt("price")
    h.newVariableInt("offered_to")  # ID of the household who made an offer
    h.newVariableInt("for_sale?")   # 0 or 1
    h.newVariableInt("for_rent?")   # 0 or 1
    h.newVariableInt("offer_confirmed?")
    h.newVariableInt("id")
    h.newVariableInt("debug_1")
    h.newVariableInt("debug_2")
    h.newVariableInt("debug_3")
    h.newVariableInt("debug_4")
    h.newVariableInt("debug_5")

    ## Households
    hh = model.newAgent("household")
    hh.newVariableInt("x")
    hh.newVariableInt("y")        
    hh.newVariableInt("income")
    hh.newVariableInt("capital")   
    hh.newVariableInt("my_house")        # ID of the house
    hh.newVariableInt("on_market?")      # 0 or 1, on the mortgage market or not
    hh.newVariableInt("move?")          # Whether the agent should physically move or not
    hh.newVariableInt("move_x")
    hh.newVariableInt("move_y")
    hh.newVariableInt("move_h")
    hh.newVariableInt("active?")         # True or False
    hh.newVariableInt("made_offer_on")
    hh.newVariableInt("monitor_id")
    hh.newVariableInt("id")
    hh.newVariableFloat("new_rand")
    hh.newVariableFloat("current_rand")
    hh.newVariableInt("debug_1")
    hh.newVariableInt("debug_2")
    hh.newVariableInt("debug_3")
    hh.newVariableInt("debug_4")
    hh.newVariableInt("debug_5")

    return h, hh

def define_model_functions(model):
    h = model.Agent("house")
    hh = model.Agent("household")  

    # Function 0.1
    hh_message_location_fn = hh.newRTCFunction(
        "hh_message_location", pyflamegpu.codegen.translate(hh_message_location)
    )
    hh_message_location_fn.setMessageOutput("hh_message_location")
    hh_message_location_fn.setMessageOutputOptional(True)

    # Function 0.2
    h_update_owner_fn = h.newRTCFunction(
        "h_update_owner", pyflamegpu.codegen.translate(h_update_owner)
    )
    h_update_owner_fn.setMessageInput("hh_message_location")
    h_update_owner_fn.setMessageOutput("h_update_owner")
    h_update_owner_fn.setMessageOutputOptional(True)

    # Function 0.3
    hh_update_ownership_fn = hh.newRTCFunction(
        "hh_update_ownership", pyflamegpu.codegen.translate(hh_update_ownership)
    )
    hh_update_ownership_fn.setMessageInput("h_update_owner")
    # hh_update_ownership_fn.setMessageOutputOptional(True)

    # Function 1
    hh_market_participation_fn = hh.newRTCFunction(
        "hh_market_participation", pyflamegpu.codegen.translate(hh_market_participation)
    )

    # Function 6
    hh_move_fn = hh.newRTCFunction(
        "hh_move", pyflamegpu.codegen.translate(hh_move)
    )
    hh_move_fn.setMessageOutput("hh_move")
    hh_move_fn.setMessageOutputOptional(True)

    # Function 7
    h_move_fn = h.newRTCFunction(
        "h_move", pyflamegpu.codegen.translate(h_move)
    )
    h_move_fn.setMessageInput("hh_move") ## Write the messages!

    

def define_submodel_functions(submodel):
    h = submodel.Agent("house")
    hh = submodel.Agent("household")

    # Function 2
    h_put_on_market_fn = h.newRTCFunction(
        "h_put_on_market", pyflamegpu.codegen.translate(h_put_on_market)
    )
    h_put_on_market_fn.setMessageOutput("h_put_on_market") ## Write the messages!
    h_put_on_market_fn.setMessageOutputOptional(True) ## Means that some houses may send the message and some may not

    # Function 3
    hh_make_offers_fn = hh.newRTCFunction(
        "hh_make_offers", pyflamegpu.codegen.translate(hh_make_offers)
    )
    hh_make_offers_fn.setMessageInput("h_put_on_market")
    hh_make_offers_fn.setMessageOutput("hh_make_offers")
    hh_make_offers_fn.setMessageOutputOptional(True)
    
    # Function 4
    h_recieve_offer_fn = h.newRTCFunction(
        "h_recieve_offer", pyflamegpu.codegen.translate(h_recieve_offer)
    )
    h_recieve_offer_fn.setMessageInput("hh_make_offers") ## Write the messages!
    h_recieve_offer_fn.setMessageOutput("h_recieve_offer") ## Write the messages!
    h_recieve_offer_fn.setMessageOutputOptional(True) ## Means that some houses may send the message and some may not

    # Function 5
    hh_offer_accepted_fn = hh.newRTCFunction(
        "hh_offer_accepted", pyflamegpu.codegen.translate(hh_offer_accepted)
    )
    hh_offer_accepted_fn.setMessageInput("h_recieve_offer") ## Write the messages!
    # hh_offer_accepted_fn.setMessageOutput("hh_offer_accepted") ## Write the messages!
    # hh_offer_accepted_fn.setMessageOutputOptional(True) ## Means that some households may send the message and some may not



## ----------------------
## Define dependencies
## ----------------------
    
def define_submodel_dependencies(submodel_model, submodel_submodel):
    h = submodel_model.Agent("house")
    hh = submodel_model.Agent("household")
    h_put_on_market_fn = h.Function("h_put_on_market")
    hh_make_offers_fn = hh.Function("hh_make_offers")
    h_recieve_offer_fn = h.Function("h_recieve_offer")
    hh_offer_accepted_fn = hh.Function("hh_offer_accepted")

    submodel_model.addExecutionRoot(h_put_on_market_fn)
    # Define dependencies
    # h_put_on_market_fn.dependsOn(hh_market_participation_fn)
    hh_make_offers_fn.dependsOn(h_put_on_market_fn)
    h_recieve_offer_fn.dependsOn(hh_make_offers_fn)
    hh_offer_accepted_fn.dependsOn(h_recieve_offer_fn)
    # hh_move_fn.dependsOn(hh_offer_accepted_fn)
    # submodel.generateLayers()

## One of the ways to print in agent functions
## math.printf()
## anything subceded with math. is dealt with as a c function

def define_model_dependencies(model, submodel):
    h = model.Agent("house")
    hh = model.Agent("household")
    hh_message_location_fn = hh.Function("hh_message_location")
    h_update_owner_fn = h.Function("h_update_owner")
    hh_update_ownership_fn = hh.Function("hh_update_ownership")
    hh_market_participation_fn = hh.Function("hh_market_participation")
    hh_move_fn = hh.Function("hh_move")
    h_move_fn = h.Function("h_move")
    

    model.addExecutionRoot(hh_message_location_fn)
    h_update_owner_fn.dependsOn(hh_message_location_fn)
    hh_update_ownership_fn.dependsOn(h_update_owner_fn)
    hh_market_participation_fn.dependsOn(hh_update_ownership_fn)
    submodel.dependsOn(hh_market_participation_fn)
    hh_move_fn.dependsOn(submodel)
    h_move_fn.dependsOn(hh_move_fn)

    model.generateLayers()

    

## ------------------
## Define messages
## ------------------

def define_messages(model):

    ## hh_message_location
    msg_hh_message_location = model.newMessageBruteForce("hh_message_location")
    msg_hh_message_location.newVariableInt("x")
    msg_hh_message_location.newVariableInt("y")
    msg_hh_message_location.newVariableInt("hh")

    ## h_update_owner
    msg_h_update_owner = model.newMessageBruteForce("h_update_owner")
    msg_h_update_owner.newVariableInt("h")
    msg_h_update_owner.newVariableInt("hh")

    ## hh_update_ownership

    ## hh_market participation

    ## h_put_on_market
    msg_h_put_on_market = model.newMessageBruteForce("h_put_on_market")
    msg_h_put_on_market.newVariableInt("h_ID")
    msg_h_put_on_market.newVariableInt("h_x")
    msg_h_put_on_market.newVariableInt("h_y")
    msg_h_put_on_market.newVariableInt("h_price")

    ## hh_make_offers
    msg_hh_make_offers = model.newMessageBruteForce("hh_make_offers")
    msg_hh_make_offers.newVariableInt("h_offered")
    msg_hh_make_offers.newVariableInt("h_offered_x")
    msg_hh_make_offers.newVariableInt("h_offered_y")
    msg_hh_make_offers.newVariableInt("hh_offering")

    ## h_recieve_offer
    msg_h_recieve_offer = model.newMessageBruteForce("h_recieve_offer")
    msg_h_recieve_offer.newVariableInt("h_offered_conf")
    msg_h_recieve_offer.newVariableInt("h_offered_conf_x")
    msg_h_recieve_offer.newVariableInt("h_offered_conf_y")
    msg_h_recieve_offer.newVariableInt("hh_offering_conf")

    ## hh_offer_accepted

    ## hh_move
    msg_hh_move = model.newMessageBruteForce("hh_move")
    msg_hh_move.newVariableInt("h_old")
    msg_hh_move.newVariableInt("h_new")
    msg_hh_move.newVariableInt("hh")

    ## h_move
    
    ## hh_immigration

    ## hh_emigration

    pass




## ------------------
##  Functions
## ------------------

@pyflamegpu.agent_function
def hh_message_location(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageBruteForce):
    # pyflamegpu.setVariableInt("id", -90)
    message_out.setVariableInt("x", pyflamegpu.getVariableInt("x"))
    message_out.setVariableInt("y", pyflamegpu.getVariableInt("y"))
    message_out.setVariableInt("hh", pyflamegpu.getID())

@pyflamegpu.agent_function
def h_update_owner(message_in: pyflamegpu.MessageBruteForce, message_out: pyflamegpu.MessageBruteForce):
    for message in message_in:
        if message.getVariableInt("x") == pyflamegpu.getVariableInt("x") and message.getVariableInt("y") == pyflamegpu.getVariableInt("y"):
            # pyflamegpu.setVariableInt("for_rent?", message.getVariableInt("hh"))
            pyflamegpu.setVariableInt("debug_1", pyflamegpu.getVariableInt("my_owner"))
            pyflamegpu.setVariableInt("my_owner", message.getVariableInt("hh"))
            pyflamegpu.setVariableInt("debug_2", pyflamegpu.getVariableInt("my_owner"))
            message_out.setVariableInt("h", pyflamegpu.getID())
            message_out.setVariableInt("hh", message.getVariableInt("hh"))

@pyflamegpu.agent_function
def hh_update_ownership(message_in: pyflamegpu.MessageBruteForce, message_out: pyflamegpu.MessageNone):
    for message in message_in:
        if message.getVariableInt("hh") == pyflamegpu.getID():
            # pyflamegpu.setVariableInt("id", message.getVariableInt("h"))
            pyflamegpu.setVariableInt("debug_1", pyflamegpu.getVariableInt("my_house"))
            pyflamegpu.setVariableInt("my_house", message.getVariableInt("h"))
            pyflamegpu.setVariableInt("debug_2", pyflamegpu.getVariableInt("my_house"))

@pyflamegpu.agent_function
def hh_market_participation(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone):
    """
    Put the household on the housing market if their capital are high
    """
    ## If I have high capital, go on the market
    pyflamegpu.setVariableInt("active?", False)
    # pyflamegpu.setVariableInt("id", pyflamegpu.getID())
    if pyflamegpu.getVariableInt("capital") > pyflamegpu.environment.getPropertyInt("CAPITAL_THRESHOLD"):
        # pyflamegpu.setVariableInt("id", -7)
        pyflamegpu.setVariableInt("on_market?", True)
        pyflamegpu.setVariableInt("active?", True)
    
    # print(pyflamegpu.getVariableInt("monitor_id"))
    return pyflamegpu.ALIVE

@pyflamegpu.agent_function
def h_put_on_market(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageBruteForce):
    """
    If the house is vacant, put it on the market
    """
    if pyflamegpu.getVariableInt("constructed?") == True and pyflamegpu.getVariableInt("offer_confirmed?") == False:
        ## If the house is vacant, put it for sale
        if pyflamegpu.getVariableInt("my_owner") == 0:
            pyflamegpu.setVariableInt("for_sale?", True)

            ## Report the house's ID
            # message_out.setIndex(pyflamegpu.getVariableInt("x"), pyflamegpu.getVariableInt("y"))
            message_out.setVariableInt("h_ID", pyflamegpu.getID())
            message_out.setVariableInt("h_x", pyflamegpu.getVariableInt("x"))
            message_out.setVariableInt("h_y", pyflamegpu.getVariableInt("y"))
            message_out.setVariableInt("h_price", pyflamegpu.getVariableInt("price"))
    return pyflamegpu.ALIVE

@pyflamegpu.agent_function
def hh_make_offers(message_in: pyflamegpu.MessageBruteForce, message_out: pyflamegpu.MessageBruteForce):
    """
    Make offers on vacant houses within my budget
    """
    if pyflamegpu.getVariableInt("active?") and pyflamegpu.getVariableInt("on_market?"):
        # current_rand = 0.0
        h_offered = 0
        h_offered_x = 0
        h_offered_y = 0
        hh_offering = 0
        n_eligible = 0.0
        for message in message_in:
            if message.getVariableInt("h_price") <= pyflamegpu.getVariableInt("capital"):
                n_eligible += 1.0
                ## Toss a die
                new_rand = pyflamegpu.random.uniformFloat()
                ## If the die value is higher than the previous highest value, make an offer on the house
                if new_rand < (1.0 / n_eligible):
                    h_offered = message.getVariableInt("h_ID")
                    h_offered_x = message.getVariableInt("h_x")
                    h_offered_y = message.getVariableInt("h_y")
                    hh_offering = pyflamegpu.getID()
                    # pyflamegpu.setVariableFloat("current_rand", current_rand)
                    # current_rand = new_rand
                    # pyflamegpu.setVariableFloat("new_rand", new_rand)

        ## If the household found a house
        if h_offered > 0:
            ## Send all the messages
            message_out.setVariableInt("h_offered", h_offered)
            message_out.setVariableInt("h_offered_x", h_offered_x)
            message_out.setVariableInt("h_offered_y", h_offered_y)
            message_out.setVariableInt("hh_offering", hh_offering)
        ## If the household found no house after going through all the messages
        else:
            ## Become inactive and send no messages
            pyflamegpu.setVariableInt("active?", False)
    return pyflamegpu.ALIVE


@pyflamegpu.agent_function
def h_recieve_offer(message_in: pyflamegpu.MessageBruteForce, message_out: pyflamegpu.MessageBruteForce):
    """
    Consider offers and send a confirmation for the accepted offer
    """
    if pyflamegpu.getVariableInt("constructed?") == True:
        n_eligible = 0.0
        h_offered_conf = 0
        h_offered_conf_x = 0
        h_offered_conf_y = 0
        hh_offering_conf = 0
        for message in message_in:
            ## If I am the house the an offer was made upon
            if message.getVariableInt("h_offered") == pyflamegpu.getID():
                n_eligible += 1.0
                ## Randomly select an offer
                new_rand = pyflamegpu.random.uniformFloat()
                if new_rand < (1.0 / n_eligible):
                    h_offered_conf = message.getVariableInt("h_offered")
                    h_offered_conf_x = message.getVariableInt("h_offered_x")
                    h_offered_conf_y = message.getVariableInt("h_offered_y")
                    hh_offering_conf = message.getVariableInt("hh_offering")
                    pyflamegpu.setVariableInt("offered_to", message.getVariableInt("hh_offering"))
        if h_offered_conf != 0:
            pyflamegpu.setVariableInt("offer_confirmed?", True)
            message_out.setVariableInt("h_offered_conf", h_offered_conf)
            message_out.setVariableInt("h_offered_conf_x", h_offered_conf_x)
            message_out.setVariableInt("h_offered_conf_y", h_offered_conf_y)
            message_out.setVariableInt("hh_offering_conf", hh_offering_conf)
        
    return pyflamegpu.ALIVE

@pyflamegpu.agent_function
def hh_offer_accepted(message_in: pyflamegpu.MessageBruteForce, message_out: pyflamegpu.MessageNone):
    """
    Updates the household `made_offer_on` to confirm the move
    """
    if pyflamegpu.getVariableInt("active?"):
        for message in message_in:
            if message.getVariableInt("hh_offering_conf") == pyflamegpu.getID():
                pyflamegpu.setVariableInt("made_offer_on", message.getVariableInt("h_offered_conf"))
                # pyflamegpu.setVariableInt("x", message.getVariableInt("h_offered_conf_x"))
                # pyflamegpu.setVariableInt("y", message.getVariableInt("h_offered_conf_y"))
                pyflamegpu.setVariableInt("move?", True)
                pyflamegpu.setVariableInt("move_x", message.getVariableInt("h_offered_conf_x"))
                pyflamegpu.setVariableInt("move_y", message.getVariableInt("h_offered_conf_y"))
                pyflamegpu.setVariableInt("move_h", message.getVariableInt("h_offered_conf"))
                pyflamegpu.setVariableInt("active?", False)
    return pyflamegpu.ALIVE

## Checks chains should be added here

@pyflamegpu.agent_function
def hh_move(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageBruteForce):
    """
    Moves the agent if it has a `made_offer_on`
    """
    pyflamegpu.setVariableInt("debug_4", 0)
    pyflamegpu.setVariableInt("debug_5", 0)
    if pyflamegpu.getVariableInt("move?") == True:
        # pyflamegpu.setVariableInt("id", -222)
        message_out.setVariableInt("h_old", pyflamegpu.getVariableInt("my_house"))
        message_out.setVariableInt("h_new", pyflamegpu.getVariableInt("move_h"))
        message_out.setVariableInt("hh", pyflamegpu.getID())

        pyflamegpu.setVariableInt("debug_4", pyflamegpu.getVariableInt("my_house"))
        pyflamegpu.setVariableInt("debug_5", pyflamegpu.getVariableInt("move_h"))
        
        pyflamegpu.setVariableInt("x", pyflamegpu.getVariableInt("move_x"))
        pyflamegpu.setVariableInt("y", pyflamegpu.getVariableInt("move_y"))
        pyflamegpu.setVariableInt("my_house", pyflamegpu.getVariableInt("move_h"))

        pyflamegpu.setVariableInt("move?", False)
        pyflamegpu.setVariableInt("move_x", 0)
        pyflamegpu.setVariableInt("move_y", 0)
        pyflamegpu.setVariableInt("move_h", 0)
        # pyflamegpu.setVariableInt("active?", False)

    return pyflamegpu.ALIVE

@pyflamegpu.agent_function
def h_move(message_in: pyflamegpu.MessageBruteForce, message_out: pyflamegpu.MessageNone):
    """
    Manages the house parameters after a household moves
    """
    pyflamegpu.setVariableInt("debug_3", 0)
    pyflamegpu.setVariableInt("debug_4", 0)
    pyflamegpu.setVariableInt("debug_5", 0)
    for message in message_in:
        if message.getVariableInt("h_old") == pyflamegpu.getID():
            pyflamegpu.setVariableInt("debug_3", -1)
            pyflamegpu.setVariableInt("my_owner", 0)
            pyflamegpu.setVariableInt("offered_to", 0)
            pyflamegpu.setVariableInt("for_sale?", False)
            pyflamegpu.setVariableInt("offer_confirmed?", False)
            # pyflamegpu.setVariableInt("id", -10)
        elif message.getVariableInt("h_new") == pyflamegpu.getID():
            pyflamegpu.setVariableInt("debug_4", pyflamegpu.getVariableInt("my_owner"))
            pyflamegpu.setVariableInt("my_owner", message.getVariableInt("hh"))
            pyflamegpu.setVariableInt("debug_5", pyflamegpu.getVariableInt("my_owner"))
            pyflamegpu.setVariableInt("offered_to", 0)
            pyflamegpu.setVariableInt("for_sale?", False)
            pyflamegpu.setVariableInt("offer_confirmed?", False)
            # pyflamegpu.setVariableInt("id", -20)

## ----------------
## Submodel
## ----------------
    
def create_submodel(model):
    active_model = pyflamegpu.ModelDescription("housing_submodel")
    # link the two models
    active_submodel = model.newSubModel("active_submodel", active_model)
    return active_model, active_submodel

def define_submodel_environment(active_model):
    define_environment(active_model)

def define_submodel_messages(active_model):
    define_messages(active_model)

def define_submodel_agents(active_model):
    define_agents(active_model)

def define_submodel_execution_order(active_model):
    # active_model.newLayer().addAgentFunction("household", "hh_market_participation")
    active_model.newLayer().addAgentFunction("house", "h_put_on_market")
    active_model.newLayer().addAgentFunction("household", "hh_make_offers")
    active_model.newLayer().addAgentFunction("house", "h_recieve_offer")
    active_model.newLayer().addAgentFunction("household", "hh_offer_accepted")
    # active_model.newLayer().addAgentFunction("household", "hh_move")
    active_model.addExitCondition(ActiveExitCondition())

def add_submodel(model):
    active_model, active_submodel = create_submodel(model)
    
    define_submodel_environment(active_model)
    define_submodel_messages(active_model)
    define_submodel_agents(active_model)
    define_submodel_functions(active_model)
    define_submodel_execution_order(active_model)

    # Note: submodel must be added to a dependency graph
    
    active_submodel.bindAgent("house", "house", True, True)
    active_submodel.bindAgent("household", "household", True, True)

    return active_model, active_submodel

# Define a host condition function
class ActiveExitCondition(pyflamegpu.HostCondition):
    def __init__(self):
        super().__init__()
        self.iterations = 0
    
    def run(self, FLAMEGPU):
        self.iterations += 1
        # Agent movements still unresolved
        
        if FLAMEGPU.agent("household").countInt("active?", True) and self.iterations < 5000:
            # print(f'Iteration {self.iterations} | active? = {FLAMEGPU.agent("household").countInt("active?", False)} | ids = {FLAMEGPU.agent("household").countInt("id", 1)}')
            return pyflamegpu.CONTINUE
        # Helpful debug statement to see how many iterations are required to resolve the conflicts
        # print(f"Conflict resolved by iteration {self.iterations}")
        self.iterations = 0
        return pyflamegpu.EXIT

## ----------------------
## Initialise model
## ----------------------

def initialise_population(
        model, 
        dimension:int=DIM, 
        density:float=DENSITY, 
        occupancy:float=OCCUPANCY, 
        price:list=PRICE, 
        capital:list=CAPITAL, 
        seed:int=SEED
        ):
    random.seed(seed)
    patches_list = pyflamegpu.AgentVector(model.Agent("house"), dimension * dimension)
    i_patch = 0
    count_occupiers = 0
    id_counter = 0

    for x, y in itertools.product(range(dimension), range(dimension)):
        ## Find the patch
        patch = patches_list[i_patch]
        id_counter += 1
        # patch.setVariableInt("id", id_counter)
        ## Allocate the patch's x and y variables
        patch.setVariableInt("x", x)
        patch.setVariableInt("y", y)
        ## Construct houses based on DENSITY
        if random.random() <= density:
            patch.setVariableInt("constructed?", True)
            patch.setVariableInt("price", random.randint(price[0], price[1]))
            ## Occupy the houses based on OCCUPANCY
            if random.random() <= occupancy:
                patch.setVariableInt("my_owner", -1)
                count_occupiers += 1
        ## Increment the patch index
        i_patch += 1

    occupiers_list = pyflamegpu.AgentVector(model.Agent("household"), count_occupiers)
    i_occupier = 0
    for i_patch in range(dimension * dimension):
        patch = patches_list[i_patch]
        if patch.getVariableInt("constructed?") == True and patch.getVariableInt("my_owner") == -1:
            occupier = occupiers_list[i_occupier]
            id_counter += 1
            # occupier.setVariableInt("id", id_counter)
            occupier.setVariableInt("x", patch.getVariableInt("x"))
            occupier.setVariableInt("y", patch.getVariableInt("y"))
            occupier.setVariableInt("my_house", patch.getID())
            occupier.setVariableInt("capital", random.randint(capital[0], capital[1]))
            patch.setVariableInt("my_owner", occupier.getID())
            i_occupier += 1

    return patches_list, occupiers_list

def initialise_simulation(
        dimension:int=DIM, 
        seed:int=SEED,
        density:float=DENSITY, 
        occupancy:float=OCCUPANCY, 
        price:list=PRICE, 
        capital:list=CAPITAL, 
        ):
    """
    Initialise the model and the population.

    Parameters
    ----------
    `dimension`: int
        Dimension of the model space (square space).
        Defaults to 50x50.
    `seed`: int
        Run seed.
        Defaults to a random number between 1 and 9999999999
    `density`: float
        The housing construction density.
        Defaults to 0.8.
    `occupancy`: float
        The occupancy rate of the houses in the system.
        Defaults to 0.8.
    `price`: list
        The range of initial prices.
        Defaults to `[300000, 300001]`
    `capital`: list
        The range of initial capital.
        Defaults to `[200000, 200001]`
    """
    global model

    model = create_model()
    define_environment(model)
    define_messages(model)
    define_agents(model)
    define_model_functions(model)
    active_model, active_submodel = add_submodel(model)
    define_submodel_dependencies(active_model, active_submodel)
    define_model_dependencies(model, active_submodel)
    # define_dependencies(model)

    # h = model.Agent("house")
    # hh = model.Agent("household")
    # h_put_on_market_fn = h.Function("h_put_on_market")

    # create an initial population of agents
    init_patches, init_occupiers = initialise_population(
        model, 
        dimension=dimension, 
        density=density,
        occupancy=occupancy,
        capital=capital,
        price=price,
        seed=seed
        )

    # create a CUDA simulation and set the initial data
    cuda_simulation = pyflamegpu.CUDASimulation(model)
    cuda_simulation.setPopulationData(init_patches)
    cuda_simulation.setPopulationData(init_occupiers)

    return cuda_simulation


def execute_simulation(
        extract_data:range|list|bool=range(SIMULATION_STEPS), 
        seed:int=SEED,
        dimension:int=DIM,
        density:float=DENSITY, 
        occupancy:float=OCCUPANCY, 
        price:list=PRICE, 
        capital:list=CAPITAL,
        simulation_steps:int=SIMULATION_STEPS
        ):
    """
    Initialise and execute the simulation.

    Parameters
    ----------
    `extract_data`: range or list or bool
        The range or list of time steps to extract the data for in `./data`.
        If `False`, no data is extracted.
        Defaults to `range(100)`.
    `dimension`: int
        Dimension of the model space (square space).
        Defaults to 50x50.
    `seed`: int
        Run seed.
        Defaults to a random number between 1 and 9999999999
    `density`: float
        The housing construction density.
        Defaults to 0.8.
    `occupancy`: float
        The occupancy rate of the houses in the system.
        Defaults to 0.8.
    `price`: list
        The range of initial prices.
        Defaults to `[300000, 300001]`
    `capital`: list
        The range of initial capital.
        Defaults to `[200000, 200001]`
    `simulation_steps`: int
        The number of simulated time steps.
        Defaults to 100.
    """
    # delete previous data
    if extract_data != False:
        if os.path.exists("data"):
            shutil.rmtree("data")
        os.mkdir("data")

    cuda_simulation = initialise_simulation(
        dimension=dimension,
        density=density,
        occupancy=occupancy,
        capital=capital,
        price=price,
        seed=seed
        )
    # Log the initial state
    if extract_data != False: cuda_simulation.exportData(f"data/0.json")
    # initialisation_data()
    print("Setup complete")
    # Set the seed
    cuda_simulation.SimulationConfig().random_seed = seed
    cuda_simulation.applyConfig()
    # simulate for number of steps and log the complete state
    for i in tqdm(range(simulation_steps)):
        cuda_simulation.step()
        if extract_data != False and i in extract_data: cuda_simulation.exportData(f"data/{i+1}.json")
    print("Simulation complete")


