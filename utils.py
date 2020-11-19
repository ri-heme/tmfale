import pandas as pd
from cobra.io import load_json_model
from pytfa import ThermoModel
from pytfa.io import load_thermoDB, read_lexicon, annotate_from_lexicon, read_compartment_data, apply_compartment_data
from pytfa.optim import DeltaG, LogConcentration
from pytfa.optim.relaxation import relax_dgo

CENTRAL_CARBON_METABOLISM = [ 'Citric Acid Cycle', 'Pentose Phosphate Pathway', 'Glycolysis/Gluconeogenesis', 'Glyoxylate Metabolism', 'Oxidative Phosphorylation', 'Pyruvate Metabolism' ]

def load_data(thermodb_path, lexicon_path, compartment_data_path):
    """
    Reads and loads necessary files to run TFA.
    """
    thermo_data = load_thermoDB( thermodb_path )
    lexicon = read_lexicon( lexicon_path )
    compartment_data = read_compartment_data( compartment_data_path )
    return thermo_data, lexicon, compartment_data

def create_model(name, cobra_path, thermo_data, lexicon, compartment_data, biomass_id):
    """
    Creates a TFA model.
    """
    cobra_model = load_json_model( cobra_path )
    tmodel = ThermoModel(thermo_data, cobra_model)
    tmodel.name = name

    annotate_from_lexicon(tmodel, lexicon)
    apply_compartment_data(tmodel, compartment_data)

    tmodel.objective = biomass_id
    tmodel.solver.problem.Params.NumericFocus = 3
    tmodel.solver.configuration.tolerances.feasibility = 1e-9
    tmodel.solver.configuration.presolve = True

    tmodel.prepare()
    tmodel.convert(verbose = False)
    
    return tmodel

def adjust_model(tmodel, lc_bounds, rxn_bounds):
    """
    Adjusts the log concentration bounds of a TFA model and knocks out the specified reaction.
    """
    # patch ATPM (non-growth associated maintenance estimated by Orth et al for P/O ratio = 1.375)
    tmodel.parent.reactions.ATPM.lower_bound = 3.15
    tmodel.reactions.ATPM.lower_bound = 3.15
    # set log concentration bounds in TFA model
    for met_id, lb, ub in zip( lc_bounds['id'], lc_bounds['lb'], lc_bounds['ub'] ):
        if tmodel.log_concentration.has_id( met_id + '_c' ):
            tmodel.log_concentration.get_by_id( met_id + '_c' ).variable.set_bounds( lb, ub )
    # constrain other reactions (e.g., growth rate, uptake/secretion rates)
    for rxn_id, lb, ub in zip( rxn_bounds['id'], rxn_bounds['lb'], rxn_bounds['ub'] ):
        if tmodel.reactions.has_id( rxn_id ):
            tmodel.parent.reactions.get_by_id(rxn_id).bounds = lb, ub
            tmodel.reactions.get_by_id(rxn_id).bounds = lb, ub

def optimize(tmodel, frisky_reactions):
    """
    Solves the LP problem and returns the (relaxed) model.
    """
    # Solve TFA
    try:
        tfa_solution = tmodel.optimize()
        relax = tfa_solution.objective_value < 0.1
    # Infeasible solutions raise this exception
    except AttributeError:
        relax = True
    # Relax dG constraints and solve again if need be
    if relax:
        relaxed_model, _, _ = relax_dgo( tmodel, reactions_to_ignore = frisky_reactions )
        relaxed_model.optimize()
        return relaxed_model
    return tmodel

def get_flux(tmodel):
    """
    Returns a 2-column DataFrame containing the calculated fluxes and the subsystem the reactions belong to.
    """
    index = pd.Index( tmodel.reactions.list_attr('id'), name = 'reaction' )
    return pd.DataFrame( [ {
        'flux': tmodel.variables[rxn.id].primal - tmodel.variables[rxn.reverse_id].primal,
        'subsystem': rxn.subsystem } for rxn in tmodel.reactions ], index = index )

def get_delta_g(tmodel):
    """
    Returns a 2-column DataFrame containing the calculated delta G and the subsystem the reactions belong to.
    """
    index = pd.Index( tmodel.delta_g.list_attr('id'), name = 'reaction' )
    subsystem = [ tmodel.reactions.get_by_id( rxn_id ).subsystem for rxn_id in index ]
    return pd.DataFrame( { 'delta_g': tmodel.get_primal(DeltaG).values, 'subsystem': subsystem }, index = index )

def get_log_concentration(tmodel):
    """
    Returns a 2-column DataFrame containing the calculated log concentration and the compartment each metabolite belongs to.
    """
    index = pd.Index( tmodel.log_concentration.list_attr('id'), name = 'metabolite' )
    compartment = [ tmodel.compartments[ tmodel.metabolites.get_by_id( met_id ).compartment ]['name'] for met_id in index ]
    return pd.DataFrame( { 'log_concentration': tmodel.get_primal(LogConcentration).values, 'compartment': compartment }, index = index )