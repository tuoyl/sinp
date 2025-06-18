import jax
import jax.numpy as jnp
import numpy as np
import re
import copy
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple
from dataclasses import dataclass
from scipy.interpolate import interp1d
from sinp.utils.refdata import load_tbabs_cross_section

__all__ = ['Model', 'Parameter', 'Component', 'JointComponent',
           'Powerlaw', 'Band', 'Brokenpowerlaw', 'TBabs']

MODEL_DICT = ["powerlaw", 'band', 'brokenpowerlaw', 'tbabs']


@dataclass
class Parameter:
    """Parameter class using dataclass for cleaner structure"""
    value: float = 0.0
    lower_lim: float = -1.0
    upper_lim: float = 1.0
    frozen: bool = False

    def __str__(self):
        return f"value: {self.value}, lower_lim: {self.lower_lim}, upper_lim: {self.upper_lim}, frozen: {self.frozen}"

    def set_par(self, parlist: List[Union[float, bool]]):
        """Set parameter from list [value, lower_lim, upper_lim, frozen]"""
        self.value = float(parlist[0])
        self.lower_lim = float(parlist[1])
        self.upper_lim = float(parlist[2])
        self.frozen = bool(parlist[3])


class ModelParams(NamedTuple):
    """Immutable container for model parameters suitable for JAX"""
    values: Dict[str, float]
    bounds: Dict[str, Tuple[float, float]]
    frozen_mask: Dict[str, bool]


class Model:
    """JAX-based SINP spectrum model"""

    def __init__(self):
        self.components = []
        self._raw_components = []
        self._input_model_str = ""
        self._joint_components = None
        self._param_dict = {}
        self._param_bounds = {}
        self._model_function = None

    def set_model(self, model_name: str):
        """Parse model name and initialize components"""
        self._input_model_str = model_name
        components = self._resolve_expression()

        if len(set(components)) != len(components):
            components_names = self._recurring_list_rename(components)
        else:
            components_names = components

        self.components = components_names
        self._raw_components = components

        # Initialize components
        for component_name, component in zip(components_names, components):
            if component.lower() in MODEL_DICT:
                component_class = self._get_component_class(component)
                component_instance = component_class()
                setattr(self, component_name, component_instance)
                setattr(component_instance, 'component_name', component_name)
            else:
                raise NameError(f"Model '{component.title()}' is not defined")

        # Create JIT-compiled model function
        self._create_model_function()

    def _get_component_class(self, component_name: str):
        """Return the appropriate component class"""
        component_map = {
            'powerlaw': Powerlaw,
            'band': Band,
            'brokenpowerlaw': Brokenpowerlaw,
            'tbabs': TBabs
        }
        return component_map[component_name.lower()]

    def get_param_dict(self) -> Dict[str, float]:
        """Get all model parameters as a flat dictionary"""
        param_dict = {}
        for component_name in self.components:
            component = getattr(self, component_name)
            for param_name in component.par_names:
                full_param_name = f"{component_name}_{param_name}"
                param_dict[full_param_name] = getattr(component, param_name).value
        return param_dict

    def get_param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for optimization"""
        bounds = {}
        for component_name in self.components:
            component = getattr(self, component_name)
            for param_name in component.par_names:
                full_param_name = f"{component_name}_{param_name}"
                param = getattr(component, param_name)
                bounds[full_param_name] = (param.lower_lim, param.upper_lim)
        return bounds

    def get_frozen_mask(self) -> Dict[str, bool]:
        """Get frozen status for all parameters"""
        mask = {}
        for component_name in self.components:
            component = getattr(self, component_name)
            for param_name in component.par_names:
                full_param_name = f"{component_name}_{param_name}"
                param = getattr(component, param_name)
                mask[full_param_name] = param.frozen
        return mask

    def get_free_params(self) -> Dict[str, float]:
        """Get only the free (non-frozen) parameters"""
        free_params = {}
        for component_name in self.components:
            component = getattr(self, component_name)
            for param_name in component.par_names:
                param = getattr(component, param_name)
                if not param.frozen:
                    full_param_name = f"{component_name}_{param_name}"
                    free_params[full_param_name] = param.value
        return free_params

    def get_free_param_names(self) -> List[str]:
        """Get names of free parameters in order"""
        return list(self.get_free_params().keys())

    def get_free_param_values(self) -> jnp.ndarray:
        """Get values of free parameters as JAX array"""
        return jnp.array(list(self.get_free_params().values()))

    def get_free_param_bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get bounds for free parameters as JAX arrays"""
        bounds = self.get_param_bounds()
        frozen_mask = self.get_frozen_mask()

        lower_bounds = []
        upper_bounds = []

        for param_name in self.get_free_param_names():
            lower, upper = bounds[param_name]
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        return jnp.array(lower_bounds), jnp.array(upper_bounds)

    def update_params(self, param_dict: Dict[str, float]):
        """Update model parameters from dictionary"""
        for full_param_name, value in param_dict.items():
            component_name, param_name = full_param_name.split('_', 1)
            if hasattr(self, component_name):
                component = getattr(self, component_name)
                if hasattr(component, param_name):
                    getattr(component, param_name).value = value

    def update_free_params(self, free_param_values: Union[jnp.ndarray, np.ndarray]):
        """Update only free parameters from array"""
        free_param_names = self.get_free_param_names()
        for i, param_name in enumerate(free_param_names):
            component_name, param_name_local = param_name.split('_', 1)
            component = getattr(self, component_name)
            getattr(component, param_name_local).value = float(free_param_values[i])

    def params_to_dict(self, free_param_values: jnp.ndarray) -> Dict[str, float]:
        """Convert free parameter array to full parameter dictionary"""
        # Start with all current parameters
        all_params = self.get_param_dict()

        # Update with free parameters
        free_param_names = self.get_free_param_names()
        for i, param_name in enumerate(free_param_names):
            #all_params[param_name] = float(free_param_values[i])
            all_params[param_name] = jnp.array((free_param_values[i]), float)

        return all_params

    def _create_model_function(self):
        """Create a JIT-compiled model function that takes a parameter array"""
        def model_fn(free_params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
            # Convert free parameters to dictionary
            param_dict = self.params_to_dict(free_params)
            return self.function(param_dict, x)

        self._model_function = jax.jit(model_fn)

    def function_free_params(self, free_params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate model with free parameters array (for optimization)"""
        param_dict = self.params_to_dict(free_params)
        return self.function(param_dict, x)

    def function(self, params: Dict[str, float], x: jnp.ndarray) -> jnp.ndarray:
        """Model function (not JIT compiled to allow dynamic parameter dict)"""
        if len(self.components) == 1:
            component = getattr(self, self.components[0])
            component_params = self._extract_component_params(params, component)
            return component.function(component_params, x)
        else:
            if self._joint_components is None:
                self._init_joint_component()
            return self._joint_components.function(params, x)

    def _extract_component_params(self, all_params: Dict[str, float], component) -> Dict[str, float]:
        """Extract parameters for a specific component"""
        component_params = {}
        prefix = f"{component.component_name}_"

        for param_name, value in all_params.items():
            if param_name.startswith(prefix):
                local_name = param_name[len(prefix):]
                component_params[local_name] = value

        return component_params

    def _init_joint_component(self):
        """Initialize joint component for multi-component models"""
        if len(self.components) == 1:
            joint_component = JointComponent(getattr(self, self.components[0]), None, 'add')
        else:
            joint_component = self._parse_and_create_joint_component()

        self._joint_components = joint_component
        return joint_component

    def _parse_and_create_joint_component(self):
        """Parse mathematical expression and create JointComponent structure"""
        model_str = self._input_model_str.replace(' ', '')

        # Create mapping from raw component names to instances
        component_mapping = {}
        for i, raw_comp in enumerate(self._raw_components):
            component_mapping[raw_comp] = getattr(self, self.components[i])

        # Tokenize expression
        tokens = []
        token_map = {}

        parts = re.split(r'([+*()])', model_str)
        parts = [p for p in parts if p]

        for part in parts:
            if part in ['+', '*', '(', ')']:
                tokens.append(part)
            elif part.lower() in [comp.lower() for comp in self._raw_components]:
                for raw_comp, instance in component_mapping.items():
                    if part.lower() == raw_comp.lower():
                        token_id = f"COMP_{len(token_map)}"
                        token_map[token_id] = instance
                        tokens.append(token_id)
                        break

        return self._evaluate_expression(tokens, token_map)

    def _evaluate_expression(self, tokens: List[str], token_map: Dict[str, Any]):
        """Evaluate tokenized expression"""
        # Handle parentheses
        while '(' in tokens:
            start = -1
            for i, token in enumerate(tokens):
                if token == '(':
                    start = i
                elif token == ')' and start != -1:
                    inner_tokens = tokens[start+1:i]
                    inner_result = self._evaluate_simple_expression(inner_tokens, token_map)
                    new_token = f"COMP_{len(token_map)}"
                    token_map[new_token] = inner_result
                    tokens = tokens[:start] + [new_token] + tokens[i+1:]
                    break

        return self._evaluate_simple_expression(tokens, token_map)

    def _evaluate_simple_expression(self, tokens: List[str], token_map: Dict[str, Any]):
        """Evaluate simple expression without parentheses"""
        if len(tokens) == 1:
            return token_map[tokens[0]]

        # Handle multiplication first
        while '*' in tokens:
            mult_idx = tokens.index('*')
            left = token_map[tokens[mult_idx-1]]
            right = token_map[tokens[mult_idx+1]]
            result = JointComponent(left, right, 'multiply')

            new_token = f"COMP_{len(token_map)}"
            token_map[new_token] = result
            tokens = tokens[:mult_idx-1] + [new_token] + tokens[mult_idx+2:]

        # Handle addition
        while '+' in tokens:
            add_idx = tokens.index('+')
            left = token_map[tokens[add_idx-1]]
            right = token_map[tokens[add_idx+1]]
            result = JointComponent(left, right, 'add')

            new_token = f"COMP_{len(token_map)}"
            token_map[new_token] = result
            tokens = tokens[:add_idx-1] + [new_token] + tokens[add_idx+2:]

        return token_map[tokens[0]]

    def _resolve_expression(self) -> List[str]:
        """Resolve model string to component list"""
        model_name = re.sub(' ', '', self._input_model_str)
        components = re.split(r"[*+()]", model_name)
        components = [x.lower() for x in components if x]
        return components

    @staticmethod
    def _recurring_list_rename(input_list: List[str]) -> List[str]:
        """Rename recurring components by adding numbers"""
        new_list = copy.copy(input_list)
        clean_list = set(input_list)

        for item in clean_list:
            indices = [i for i, x in enumerate(input_list) if x == item]
            if len(indices) > 1:
                for nth_item, index in enumerate(indices):
                    new_list[index] = f"{input_list[index]}{nth_item+1}"

        if new_list != input_list:
            warnings.warn(f"CAUTION: Components renamed! {input_list} -> {new_list}")

        return new_list

    def spectrum(self, energy: jnp.ndarray) -> jnp.ndarray:
        """Calculate photon spectrum"""
        params = self.get_param_dict()
        return self.function(params, energy)

    def show(self):
        """Display model components and parameters"""
        for component_name in self.components:
            component = getattr(self, component_name)
            print(component)


class Component:
    """Base component class for JAX-based spectral models"""

    def __init__(self):
        self.nparams = 1
        self.component_name = ''
        self._par_names = []

    def __str__(self):
        string = f"Model Component <{self.component_name}>:\n\tParameters:\n"
        for par in self.par_names:
            string += f"\t\t{par}: {{{getattr(self, par)}}}\n"
        return string

    @property
    def nparams_free(self) -> int:
        """Count of free parameters"""
        return sum(1 for param_name in self._par_names
                  if not getattr(self, param_name).frozen)

    @property
    def par_names(self) -> List[str]:
        return self._par_names

    @par_names.setter
    def par_names(self, parlist: List[str]):
        self._par_names = parlist
        self.nparams = len(parlist)
        for parameter in parlist:
            setattr(self, parameter, Parameter())

    @property
    def par_values(self) -> List[float]:
        """Get all parameter values"""
        return [getattr(self, param).value for param in self.par_names]

    @property
    def par_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds"""
        return [(getattr(self, param).lower_lim, getattr(self, param).upper_lim)
                for param in self.par_names]

    @property
    def par_frozen_flags(self) -> List[bool]:
        """Get frozen flags"""
        return [getattr(self, param).frozen for param in self.par_names]

    def function(self, params: Dict[str, float], x: jnp.ndarray) -> jnp.ndarray:
        """Component function - to be implemented by subclasses"""
        raise NotImplementedError

    def __add__(self, other):
        return JointComponent(self, other, 'add')

    def __mul__(self, other):
        return JointComponent(self, other, 'multiply')


class JointComponent:
    """Joint component for combining multiple components"""

    def __init__(self, component1, component2, expression: str):
        self._expression = expression
        self._component1 = component1
        self._component2 = component2

        if component2 is None:
            self.component_list = [component1.component_name]
            self.nparams = component1.nparams
        else:
            self._setup_two_components(component1, component2)

    def _setup_two_components(self, component1, component2):
        """Setup for two-component combination"""
        # Handle component lists
        if isinstance(component1, JointComponent):
            self.component_list = component1.component_list.copy()
        else:
            self.component_list = [component1.component_name]

        if isinstance(component2, JointComponent):
            self.component_list.extend(component2.component_list)
        else:
            self.component_list.append(component2.component_name)

        self.nparams = self._get_nparams(component1) + self._get_nparams(component2)

    def _get_nparams(self, component) -> int:
        """Get number of parameters"""
        return component.nparams

    def function(self, params: Dict[str, float], x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate joint component function"""
        if self._component2 is None:
            return self._component1.function(
                self._extract_component_params(params, self._component1), x)

        result1 = self._component1.function(
            self._extract_component_params(params, self._component1), x)
        result2 = self._component2.function(
            self._extract_component_params(params, self._component2), x)

        if self._expression == 'add':
            return result1 + result2
        elif self._expression == 'multiply':
            return result1 * result2
        else:
            raise ValueError(f"Unknown expression: {self._expression}")

    def _extract_component_params(self, all_params: Dict[str, float], component) -> Dict[str, float]:
        """Extract parameters for a component"""
        if isinstance(component, JointComponent):
            return all_params
        else:
            component_params = {}
            prefix = f"{component.component_name}_"

            for param_name, value in all_params.items():
                if param_name.startswith(prefix):
                    local_name = param_name[len(prefix):]
                    component_params[local_name] = value

            return component_params

    def __add__(self, other):
        return JointComponent(self, other, 'add')

    def __mul__(self, other):
        return JointComponent(self, other, 'multiply')


class Powerlaw(Component):
    """JAX-based Powerlaw component"""

    def __init__(self):
        super().__init__()
        self.component_name = 'powerlaw'
        self.par_names = ['norm', 'photon_index']
        self.norm.set_par([10, 0, 1e6, False])
        self.photon_index.set_par([2.1, 0, 10, False])

    def function(self, params: Dict[str, float], x: jnp.ndarray) -> jnp.ndarray:
        """Powerlaw: A(E) = K * E^(-alpha)"""
        norm = params['norm']
        photon_index = params['photon_index']
        return norm * jnp.power(x, -photon_index)


class Band(Component):
    """JAX-based Band GRB function"""

    def __init__(self):
        super().__init__()
        self.component_name = "band"
        self.par_names = ['norm', 'Epeak', 'alpha', 'beta', 'Epiv']
        self.norm.set_par([0.01, 1e-10, np.inf, False])
        self.Epeak.set_par([500, 0.01, np.inf, False])
        self.alpha.set_par([-0.5, -1.9, 20, False])
        self.beta.set_par([-2.5, -10, -2, False])
        self.Epiv.set_par([100, 0.01, np.inf, False])

    def function(self, params: Dict[str, float], x: jnp.ndarray) -> jnp.ndarray:
        """Band spectrum function"""
        norm = params['norm']
        epeak = params['Epeak']
        alpha = params['alpha']
        beta = params['beta']
        epiv = params['Epiv']

        # Band function implementation
        ebreak = epeak * (alpha - beta) / (2 + alpha)

        # Low energy part
        low_energy = norm * jnp.power(x / epiv, alpha) * jnp.exp(-x * (2 + alpha) / epeak)

        # High energy part
        high_energy = (norm * jnp.power(ebreak / epiv, alpha - beta) *
                      jnp.exp(beta - alpha) * jnp.power(x / epiv, beta))

        return jnp.where(x <= ebreak, low_energy, high_energy)


class Brokenpowerlaw(Component):
    """JAX-based Broken powerlaw component"""

    def __init__(self):
        super().__init__()
        self.component_name = "brokenpowerlaw"
        self.par_names = ['norm', 'Ebreak', 'alpha', 'beta', 'Epiv']
        self.norm.set_par([0.01, 1e-10, np.inf, False])
        self.Ebreak.set_par([700, 0.01, np.inf, False])
        self.alpha.set_par([1.0, -20, 2.0, False])
        self.beta.set_par([2.0, -20, 10, False])
        self.Epiv.set_par([100, 0.01, np.inf, True])

    def function(self, params: Dict[str, float], x: jnp.ndarray) -> jnp.ndarray:
        """Broken powerlaw function"""
        norm = params['norm']
        ebreak = params['Ebreak']
        alpha = params['alpha']
        beta = params['beta']
        epiv = params['Epiv']

        low_energy = jnp.power(x / epiv, -alpha)
        high_energy = (jnp.power(ebreak / epiv, -alpha) *
                      jnp.power(x / ebreak, -beta))

        return norm * jnp.where(x <= ebreak, low_energy, high_energy)


class TBabs(Component):
    """JAX-based TBabs absorption model"""

    def __init__(self):
        super().__init__()
        self.component_name = "tbabs"
        self.par_names = ['nH']
        self.nH.set_par([0.05, 0.0, 10.0, False])

        # Load cross-section data and create interpolation function
        energy, sigma = load_tbabs_cross_section()
        self._energy_data = energy
        self._sigma_data = sigma

        # For JAX compatibility, we'll use a simple linear interpolation
        # that can be JIT compiled
        self._log_energy = np.log(energy)
        self._log_sigma = np.log(sigma)

    def _interpolate_sigma(self, x: jnp.ndarray) -> jnp.ndarray:
        """JAX-compatible interpolation of cross-section"""
        # Convert to JAX arrays if not already
        log_energy_jax = jnp.array(self._log_energy)
        log_sigma_jax = jnp.array(self._log_sigma)

        log_x = jnp.log(x)

        # JAX-compatible interpolation
        # For each x value, find the interpolation indices
        def interpolate_single(log_x_val):
            # Find index using JAX operations
            idx = jnp.searchsorted(log_energy_jax, log_x_val)
            idx = jnp.clip(idx, 1, len(log_energy_jax) - 1)

            # Get interpolation points
            x0 = log_energy_jax[idx - 1]
            x1 = log_energy_jax[idx]
            y0 = log_sigma_jax[idx - 1]
            y1 = log_sigma_jax[idx]

            # Linear interpolation
            t = (log_x_val - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

        # Vectorize over all x values
        log_sigma_interp = jax.vmap(interpolate_single)(log_x)


        return jnp.exp(log_sigma_interp)

    def function(self, params: Dict[str, float], x: jnp.ndarray) -> jnp.ndarray:
        """TBabs absorption: A(E) = exp(-nH * sigma(E))"""
        nH = params['nH']
        sigma_E = self._interpolate_sigma(x)
        return jnp.exp(-nH * sigma_E)
