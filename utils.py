import pandas as pd
from cobra.exceptions import Infeasible
from cobra.io import load_json_model
from pytfa import ThermoModel
from pytfa.io import load_thermoDB, read_lexicon, annotate_from_lexicon, read_compartment_data, apply_compartment_data
from pytfa.optim import DeltaG, LogConcentration
from pytfa.optim.relaxation import relax_dgo
from pytfa.utils.numerics import BIGM

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
    Adjusts the log concentration and flux bounds of a TFA model.
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

def optimize(tmodel, relax = 'dgo', frisky_vars = []):
    """
    Relaxes and solves the LP problem. Returns the relaxed model and a table with the magnitude of the bound violations.
    """
    if relax == 'dgo':
        relaxed_model, _, relax_table = relax_dgo( tmodel, reactions_to_ignore = frisky_vars )
        relaxed_model.optimize()
        return relaxed_model, get_dgo_bound_change( relaxed_model, relax_table )
    elif relax == 'lc':
        return relax_lc( tmodel, metabolites_to_ignore = frisky_vars )
    raise ValueError('Only standard Gibbs energy (`dgo`) or log concentration (`lc`) relaxation allowed.')

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

def get_dgo_bound_change(tmodel, relax_table):
    """
    Returns a 2-column DataFrame containing the reaction bound changes and the subsystem their reaction belongs to.
    """
    bound_change = relax_table['ub_change'] - relax_table['lb_change']
    relax_table.index.name = 'reaction'
    return pd.DataFrame( {
        'bound_change': bound_change.values,
        'subsystem': tmodel.reactions.query( lambda rxn_id : rxn_id in relax_table.index, 'id' ).list_attr( 'subsystem' ) },
        index = relax_table.index )

def relax_lc(tmodel, relax_obj_type = 0, metabolites_to_ignore = [], destructive = True):
    """
    Uses the Gurobi subroutines to relax the log concentration bounds. Returns a table with the magnitude of the bound violations.
    """
    if tmodel.solver.interface.__name__ != 'optlang.gurobi_interface':
        raise ModuleNotFoundError('Requires Gurobi.')

    # copy Gurobi model
    grb_model = tmodel.solver.problem.copy()
    
    # get the Gurobi log concentration variables
    lc_vars = [ grb_model.getVarByName( var.name ) for var in tmodel.log_concentration if  var.id not in metabolites_to_ignore ]
    vars_penalties = [1] * len(lc_vars)

    # perform relaxation of variable bounds
    relax_obj = grb_model.feasRelax(relaxobjtype = relax_obj_type,
                                    minrelax = True,
                                    vars = lc_vars,
                                    lbpen = vars_penalties,
                                    ubpen = vars_penalties,
                                    constrs = None,
                                    rhspen = None)
    
    # check if relaxation was successful
    if relax_obj < 0:
        raise Infeasible('Failed to create the feasibility relaxation!')
        
    grb_model.optimize()
    
    # transfer the bound changes from Gurobi Model to ThermoModel
    rows = []
    metabolites = tmodel.log_concentration.list_attr('id')
    for met_id in metabolites:
        # get the auxiliary LB/UB change variables
        lb_change = -grb_model.getVarByName('ArtL_LC_' + met_id).X
        ub_change = grb_model.getVarByName('ArtU_LC_' + met_id).X
        
        # check if the auxiliary variables are not 0 (indicating a bound change)
        if lb_change < 0 or ub_change > 0:
            
            # set new bounds on the original log concentration variable
            if destructive:
                var = tmodel.log_concentration.get_by_id(met_id).variable
                var.set_bounds( lb = max( -BIGM, min( BIGM, var.lb + lb_change ) ),
                                ub = max( -BIGM, min( BIGM, var.ub + ub_change ) ) )
            
            rows.append( { 'metabolite': met_id, 'bound_change': lb_change + ub_change,
                'compartment': tmodel.compartments[ tmodel.metabolites.get_by_id( met_id ).compartment ]['name'] } )
    
    tmodel.optimize()
    
    # return table of relaxed variables
    return pd.DataFrame.from_records( rows, index = 'metabolite' )