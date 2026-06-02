"""gymansium RL environment for IBP reduction of a simple family of Feynman integrals,
i.e. massive bubble integrals.
"""

from juliacall import Main as jl
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import matplotlib as plt
from matplotlib.animation import FuncAnimation
import random
from typing import Sequence, List, Tuple
from functools import cmp_to_key
import re

jl.seval("import FeynGym")
jl.seval("import SparseSolveExact")
jl_feyngym = jl.FeynGym
jl_solver = jl.SparseSolveExact

N_CHANNELS = jl_feyngym.actual_data_length_per_integral
MASK_N_CHANNELS = jl_feyngym.mask_data_length_per_integral
DATA_N_CHANNELS = jl_feyngym.actual_data_length_per_integral

# def print_params():
#     """Print the kinematic and dimension parameters used for the Feynman integral"""
#     print(f"Parameters: {p2=} , {m2=} , {d=}")


class BubbleEnv:
    """Thin wrapper around the Julia package FeynGym.jl"""

    def __init__(
        self,
        target_integral=[6, 6],
        skip_redundant_equation_cost=False,
        max_seed_propagator_power=6,
        random_target=False,
        normalize_reward=True,
    ):
        self.target_integral = target_integral
        self.skip_redundant_equation_cost = skip_redundant_equation_cost
        self.max_seed_propagator_power = max_seed_propagator_power
        self.random_target = random_target
        self.normalize_reward = normalize_reward
        if random_target:
            self.actual_target_integral = [
                random.randint(2, max_seed_propagator_power),
                random.randint(2, max_seed_propagator_power),
            ]
        else:
            self.actual_target_integral = target_integral
        self.jl_env = jl_feyngym.BubbleEnv(
            self.actual_target_integral,
            skip_redundant_equation_cost,
            max_seed_propagator_power,
            "square",
        )

    def act(self, a):
        """Take an action and return the reward, as the negative of the cost,
        possiby normalized by a factor"""
        cost = jl_feyngym.act_b(self.jl_env, a)
        if self.normalize_reward:
            return -cost / (
                self.actual_target_integral[0] * self.actual_target_integral[1]
            )
        else:
            return -cost

    def actions(self):
        """Return the list of actions allowed by the environtment"""
        return np.array(jl_feyngym.actions1(self.jl_env))

    def store_observation_and_mask(self):
        """Store observation and mask data"""
        obs0, mask0 = jl_feyngym.observation_and_mask_2d(self.jl_env)
        self.obs = np.asarray(obs0)
        self.mask = np.asarray(mask0).flatten()

    def terminated(self):
        """Whether the environment has terminated"""
        return jl_feyngym.terminated(self.jl_env)

    def reset(self):
        """Reset the environment"""
        if self.random_target:
            self.actual_target_integral = [
                random.randint(2, self.max_seed_propagator_power),
                random.randint(2, self.max_seed_propagator_power),
            ]
            self.jl_env = jl_feyngym.BubbleEnv(
                self.actual_target_integral,
                self.skip_redundant_equation_cost,
                self.max_seed_propagator_power,
                "square",
            )
        else:
            jl_feyngym.reset_b(self.jl_env)

    def action_log(self):
        """Return the list of actions taken since the initializatio of the environment"""
        return np.asarray(jl_feyngym.action_log1(self.jl_env))


class IBPEnv(gym.Env):
    """RL environemnt in the gymnasium interface for bubble integrals"""

    def __init__(
        self,
        render_mode=None,
        truncate_if_invalid=False,
        target_integral=[6, 6],
        skip_redundant_equation_cost=False,
        max_seed_propagator_power=6,
        random_target=False,
        normalize_reward=True,
    ):
        self.state = BubbleEnv(
            target_integral,
            skip_redundant_equation_cost,
            max_seed_propagator_power,
            random_target,
            normalize_reward,
        )
        self.all_actions_2d = np.array(
            self.state.jl_env.properties.all_actions_2d
        ).flatten()
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(
                jl_feyngym.actual_data_length_per_integral,
                max_seed_propagator_power + 3,
                max_seed_propagator_power + 3,
            ),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(
            (jl_feyngym.n_ibp_operators + 1) * (max_seed_propagator_power + 3) ** 2
        )
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.state.store_observation_and_mask()
        self.valid_actions = self.define_valid_actions()

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        self.state.reset()
        super().reset(seed=seed)
        self.state.store_observation_and_mask()
        observation = self._get_obs()
        info = self._get_info()
        self.valid_actions = self.define_valid_actions()
        return observation, info

    def _get_obs(self):
        """Return the observations"""
        return self.state.obs

    def _get_info(self):
        """Return information about the environemnt, in particular the action masks"""
        return {"action_mask": self.action_masks()}

    def define_valid_actions(self):
        """Return the set of valid actions, as integer indices in reference to the list of all actions"""
        return np.nonzero(self.action_masks().flatten())[0]

    def action_masks(self):
        """Return the action mask"""
        return self.state.mask

    def step(self, action):
        """Take one step of actions"""
        # print(f"step with {action}")
        action_raw = self.all_actions_2d[action]
        reward = self.state.act(action_raw)
        self.state.store_observation_and_mask()
        observation = self._get_obs()
        terminated = self.state.terminated()
        truncated = False
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def action_log(self):
        """Return the list of actions taken since the initializatio of the environment"""
        return self.state.action_log()

    def visualize(self):
        board_size = self.state.max_seed_propagator_power + 3
        return visualize_board(
            self, self.state.target_integral, (board_size, board_size)
        )

    def render(self):
        pass

    def close(self):
        pass

    def all_seed_candidates(self):
        all_actions = self.state.jl_env.properties.all_possibly_valid_actions
        all_seeds = list(set((s[0], s[1]) for s in all_actions if s[2] > 0))
        all_seeds.sort(key=lambda s: s[0] + s[1])
        return all_seeds


# Register the environment with gymnasium
register(
    id="pyfeyngym-v0",
    entry_point="pyfeyngym:IBPEnv",
    max_episode_steps=None,
)


def _test_env(target_integral=[6, 6], max_seed_propagator_power=6):
    e = IBPEnv(
        target_integral=target_integral,
        max_seed_propagator_power=max_seed_propagator_power,
    )
    # e = gym.make("pyfeyngym-v0", target_integral=target_integral, max_seed_propagator_power=max_seed_propagator_power)
    total_reward = 0.0
    n_steps = 0
    while True:
        a = e.define_valid_actions()[0]
        print(f"taking action {a}: {e.all_actions_2d[a]}")
        _, re, term, _, _ = e.step(a)
        print(f"reward = {re}")
        total_reward += re
        n_steps += 1
        if term:
            break
    print(f"{total_reward = }")
    print(f"{n_steps = }")


import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, Polygon


def visualize_board(env: IBPEnv, target_integral, board_size, delay_ms=100):
    """Visualize the environment, particularly all the actions taken (which constitute the observation)"""
    actions_taken = env.action_log()
    seeds = [(int(a[0]), int(a[1]), int(a[2])) for a in actions_taken if a[2] != 0]
    return visualize_seeds(seeds, target_integral, board_size, delay_ms)


def visualize_seeds(seeds, target_integral, board_size, delay_ms):
    """
    Internal visualization method called by `visualize_board`. Arguments:
    seeds: list of tuples (x, y, t)
      - (x, y) are integer center coordinates of a square
      - t is 1 or 2:
          1 -> draw a triangle on the left half of the square
          2 -> draw a triangle on the right half of the square

    The returned axes includes a legend for both IBP operators and the
    highlighted target integral.
    """
    fig, ax = plt.subplots(figsize=(board_size[0], board_size[1]))

    ax.add_patch(
        Rectangle(
            (target_integral[0] - 0.5, target_integral[1] - 0.5),
            1,
            1,
            fill=False,
            edgecolor="red",
            linewidth=2,
            zorder=10,
        )
    )

    seed_polys = []
    # Draw squares centered at integer coordinates
    for x in range(-1, board_size[0] - 1):
        for y in range(-1, board_size[1] - 1):
            # gray out the squares corresponding to integrals forbidden as seeds
            gray = (
                (x < 0)
                or (y < 0)
                or (x > board_size[0])
                or (y > board_size[1])
                or (x == y == 0)
            )
            ax.add_patch(
                Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    edgecolor="black",
                    linewidth=1,
                    facecolor="lightgray" if gray else "none",
                )
            )

    # Add seed triangles
    for sx, sy, t in seeds:
        if t == 1:  # left half: triangle from left edge pointing to center
            verts = [
                (sx - 0.1 - 0.25, sy - 0.25),
                (sx - 0.1 - 0.25, sy + 0.25),
                (sx - 0.1, sy),
            ]
            color = "tab:orange"
        elif t == 2:  # right half: triangle from right edge pointing to center
            verts = [
                (sx + 0.1 + 0.25, sy - 0.25),
                (sx + 0.1 + 0.25, sy + 0.25),
                (sx + 0.1, sy),
            ]
            color = "tab:blue"
        else:
            raise ValueError("Seed third integer must be 1 (left) or 2 (right).")
        poly = Polygon(
            verts, closed=True, facecolor=color, edgecolor="black", linewidth=1
        )
        ax.add_patch(poly)
        seed_polys.append(poly)

    # Set ticks at the centers of the squares
    ax.set_xticks(range(-1, board_size[0] - 1))
    ax.set_yticks(range(-1, board_size[1] - 1))

    # Limits so the outer squares are fully visible
    ax.set_xlim(-1.5, board_size[0] - 1.5)
    ax.set_ylim(-1.5, board_size[1] - 1.5)

    ax.set_aspect("equal", adjustable="box")
    x_label = ax.set_xlabel(r'$a_1$')
    y_label = ax.set_ylabel(r'$a_2$')
    doubled_label_size = x_label.get_size() * 2
    x_label.set_size(doubled_label_size)
    y_label.set_size(doubled_label_size)
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.14))
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 2)
    legend = ax.legend(
        handles=[
            Patch(
                facecolor="tab:orange",
                edgecolor="black",
                label="IBP operator 1",
            ),
            Patch(
                facecolor="tab:blue",
                edgecolor="black",
                label="IBP operator 2",
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                fill=False,
                edgecolor="red",
                linewidth=2,
                label=f"target=({target_integral[0]:g}, {target_integral[1]:g})",
            ),
        ],
        loc="best",
    )
    for text in legend.get_texts():
        text.set_fontsize(text.get_fontsize() * 2)

    # Animation: reveal triangles one by one
    def init():
        for p in seed_polys:
            p.set_visible(False)
        return seed_polys

    def update(i):
        seed_polys[i].set_visible(True)
        return [seed_polys[i]]

    if len(seed_polys) > 0:
        anim = FuncAnimation(
            fig,
            update,
            frames=len(seed_polys),
            init_func=init,
            interval=delay_ms,
            blit=False,
            repeat=False,
        )
    else:
        anim = None
    return fig, ax, anim


def test_ibp_with_seed_integrals(
    target_integral=[6, 6],
    skip_redundant_equation_cost=False,
    max_seed_propagator_power=6,
    seeds=[],
    failure_cost=10**6,
):

    env = IBPEnv(
        target_integral=target_integral,
        skip_redundant_equation_cost=skip_redundant_equation_cost,
        max_seed_propagator_power=max_seed_propagator_power,
    )
    cost = jl_feyngym.act_automatic_with_seeds_b(env.state.jl_env, seeds)
    if not env.state.terminated():
        cost = failure_cost
    return cost


def test_ibp_with_priority_function(
    priority_function,
    target_integral=[6, 6],
    skip_redundant_equation_cost=False,
    max_seed_propagator_power=6,
    return_finished_env=False,
):
    priority_list = compute_priority_list(
        priority_function,
        target_integral=target_integral,
        skip_redundant_equation_cost=skip_redundant_equation_cost,
        max_seed_propagator_power=max_seed_propagator_power,
    )
    return test_ibp_with_priority_list(
        priority_list,
        target_integral=target_integral,
        skip_redundant_equation_cost=skip_redundant_equation_cost,
        max_seed_propagator_power=max_seed_propagator_power,
        return_finished_env=return_finished_env,
    )


def compute_priority_list(
    priority_function,
    target_integral=[6, 6],
    skip_redundant_equation_cost=False,
    max_seed_propagator_power=6,
):
    all_actions = get_action_list(max_seed_propagator_power=max_seed_propagator_power)
    priority_list = [priority_function(a) for a in all_actions]
    return priority_list


def get_action_list(max_seed_propagator_power=6):
    env = IBPEnv(
        target_integral=[max_seed_propagator_power, max_seed_propagator_power],
        max_seed_propagator_power=max_seed_propagator_power,
    )
    all_actions = np.array([np.array(a) for a in env.all_actions_2d])
    return all_actions


def test_ibp_with_priority_list(
    priority_list,
    target_integral=[6, 6],
    skip_redundant_equation_cost=False,
    max_seed_propagator_power=6,
    return_finished_env=False,
):
    env = IBPEnv(
        target_integral=target_integral,
        skip_redundant_equation_cost=skip_redundant_equation_cost,
        max_seed_propagator_power=max_seed_propagator_power,
    )
    terminated = False
    total_reward = 0
    while not terminated:
        valid_action_indices = env.define_valid_actions()
        priority_scores = [priority_list[j] for j in valid_action_indices]
        highest_score_index = int(np.argmax(priority_scores))
        chosen_action_index = valid_action_indices[highest_score_index]
        observation, reward, terminated, truncated, info = env.step(chosen_action_index)
        total_reward += reward
    if not return_finished_env:
        return total_reward
    else:
        return total_reward, env


def solve_eqs_modulo(
    eqs: List[Sequence],
    variables: List,
    modulus: int,
    *,
    return_info=False,
    keep_on_rhs=None,
    complete_pivoting=False,
    naive_pivoting=False,
    nopivoting=False,
    needed_variables: set = None,
):
    if return_info:
        jl_solution, cost, eqs_in_order, vars_in_order = jl_solver.solve_eqs_modulo(
            eqs,
            variables,
            modulus,
            return_info=True,
            keep_on_rhs=keep_on_rhs,
            complete_pivoting=complete_pivoting,
            naive_pivoting=naive_pivoting,
            nopivoting=nopivoting,
        )
    else:
        jl_solution = jl_solver.solve_eqs_modulo(
            eqs,
            variables,
            modulus,
            keep_on_rhs=keep_on_rhs,
            complete_pivoting=complete_pivoting,
            naive_pivoting=naive_pivoting,
            nopivoting=nopivoting,
        )
    if needed_variables is None:
        solution_dict = {r[0]: [[t[0], t[1]] for t in r[1]] for r in jl_solution}
    else:
        solution_dict = {
            r[0]: [[t[0], t[1]] for t in r[1]]
            for r in jl_solution
            if r[0] in needed_variables
        }
    if return_info:
        return solution_dict, cost, eqs_in_order, vars_in_order
    else:
        return solution_dict


# Generating IBP equations from Kira files for equation templates and trivial sectors
def gen_eq_templates(filename, m_values=None):
    """
    Parse a file with triangular equations.

    Args:
        filename: Path to the input file
        m_values: Dictionary mapping m variable names to their values,
                  e.g., {'m1': 2, 'm2': 3, 'm3': 4} or list [d, m1, m2, m3, ...]
                  If a list is provided, it should be [d_val, m1_val, m2_val, m3_val, ...]
    """
    if m_values is None:
        m_values = {}
    elif isinstance(m_values, list):
        # Convert list to dictionary: [d, m1, m2, m3, ...] -> {'d': val, 'm1': val, ...}
        m_dict = {}
        if len(m_values) > 0:
            m_dict["d"] = m_values[0]
        for i in range(1, len(m_values)):
            m_dict[f"m{i}"] = m_values[i]
        m_values = m_dict

    with open(filename, "r") as f:
        content = f.read()

    # Split by double newlines to separate equation groups
    equation_groups = content.strip().split("\n\n")

    result = []

    for idx, group in enumerate(equation_groups, 1):
        equations = [line.strip() for line in group.strip().split("\n") if line.strip()]

        group_data = []

        for equation in equations:
            # Parse the equation: anything[...]*(...)
            match = re.match(r".*?\[(.*?)\]\s*\*\s*\((.*?)\)", equation)
            if not match:
                continue

            # Extract the vector inside brackets
            vector_str = match.group(1)
            vector = [int(x.strip()) for x in vector_str.split(",")]

            # Extract the coefficient expression
            coef_expr = match.group(2).replace("^", "**")

            # Replace m1, m2, m3, etc. with their values (wrapped in parentheses for safety)
            for m_name, m_val in m_values.items():
                if m_name not in ("d",) and not m_name.startswith("a"):
                    # Replace m1, m2, m3 with their numerical values
                    coef_expr = re.sub(r"\b" + m_name + r"\b", f"({m_val})", coef_expr)

            # Build coefficient vector for a0, a1, ..., a{len(vector)-1}, d
            coef_vector = []

            # Check each ai coefficient
            for i in range(len(vector)):
                coef_name = f"a{i}"
                # Split expression by + and - while keeping the operators
                terms = re.split(r"(\+|-)", coef_expr)

                coefficient = 0
                current_sign = 1

                for term in terms:
                    term = term.strip()
                    if term == "+":
                        current_sign = 1
                    elif term == "-":
                        current_sign = -1
                    elif term:
                        # Check if this term contains the coefficient we're looking for
                        if re.search(r"\b" + coef_name + r"\b", term):
                            # Remove the coefficient name to get just the numerical part
                            # Replace ai with *1* to preserve multiplication structure
                            numerical_part = re.sub(
                                r"\b" + coef_name + r"\b", "*1*", term
                            )
                            # Clean up multiple asterisks
                            numerical_part = re.sub(r"\*+", "*", numerical_part)
                            # Remove leading/trailing asterisks
                            numerical_part = numerical_part.strip("*").strip()

                            if numerical_part == "" or numerical_part == "1":
                                coefficient += current_sign * 1
                            else:
                                # Evaluate the numerical expression (now properly handles multiplication)
                                try:
                                    coefficient += current_sign * eval(numerical_part)
                                except Exception as e:
                                    print(f"Error evaluating '{numerical_part}': {e}")
                                    coefficient += current_sign * 1

                coef_vector.append(coefficient)

            # Check for 'd' coefficient
            terms = re.split(r"(\+|-)", coef_expr)

            d_coefficient = 0
            current_sign = 1

            for term in terms:
                term = term.strip()
                if term == "+":
                    current_sign = 1
                elif term == "-":
                    current_sign = -1
                elif term:
                    # Check if this term contains 'd'
                    if re.search(r"\bd\b", term):
                        # Remove 'd' to get just the numerical part
                        # Replace d with *1* to preserve multiplication structure
                        numerical_part = re.sub(r"\bd\b", "*1*", term)
                        # Clean up multiple asterisks
                        numerical_part = re.sub(r"\*+", "*", numerical_part)
                        # Remove leading/trailing asterisks
                        numerical_part = numerical_part.strip("*").strip()

                        if numerical_part == "" or numerical_part == "1":
                            d_coefficient += current_sign * 1
                        else:
                            # Evaluate the numerical expression
                            try:
                                d_coefficient += current_sign * eval(numerical_part)
                            except Exception as e:
                                print(f"Error evaluating '{numerical_part}': {e}")
                                d_coefficient += current_sign * 1

            coef_vector.append(d_coefficient)

            group_data.append([vector, coef_vector])

        result.append((idx, group_data))

    return result


def get_trivial_sectors(filename, cut: list[int] = [], n_indices: int = 7):
    with open(filename, "r") as f:
        content = f.read().strip()
    loaded_sectors = [int(x) for x in content.split(",") if x.strip()]
    if cut:
        trivial_sectors_from_cut = [
            s
            for s in range(2**n_indices)
            if not pass_cut(to_sector_list(s, padding=n_indices), cut)
        ]
    else:
        trivial_sectors_from_cut = []
    all_trivial_sectors = sorted(
        list(set(loaded_sectors) | set(trivial_sectors_from_cut))
    )
    return all_trivial_sectors


def vdot(vec1, vec2):
    # print(vec1)
    # print(vec2)
    return sum(a * b for a, b in zip(vec1, vec2))


def vadd(vec1, vec2):
    return tuple([a + b for a, b in zip(vec1, vec2)])


def to_sector(a_j):
    """
    Computes the sum {j=1...n} 2^(j-1) * theta(a_j),
    where theta is the Heaviside step function.

    Parameters:
        *a_j: A variable number of arguments representing a_j values.

    Returns:
        The computed sum as an integer.
    """
    # Heaviside step function (theta)
    theta = lambda x: 1 if x > 0 else 0
    # Compute the sum
    sectornumber = 0
    for j, a in enumerate(a_j, start=1):
        sectornumber += (2 ** (j - 1)) * theta(a)
    # print(a_j)
    # print(sectornumber)
    return sectornumber


def to_sector_list(sector_num: int, padding) -> list[int]:
    """
    Inverse of to_sector.
    Given a positive integer sector_num, return a list of 0/1 where
    the j-th element (0-based) corresponds to the bit of weight 2^j
    (least-significant bit first).
    """
    if sector_num <= 0:
        return []
    bits: list[int] = []
    while sector_num:
        bits.append(sector_num & 1)
        sector_num >>= 1
    # Pad with zeros to ensure the list has the desired length
    while len(bits) < padding:
        bits.append(0)
    return bits


def pass_cut(sector_list: list[int], cut: list[int]) -> bool:
    """
    Check if a sector passes the cut.
    The cut is a list of integers indicating which propagators to cut (1-based).
    A sector fails the cut if it has a non-negative value in any position corresponding to a cut propagator.
    """
    for i in cut:
        if i - 1 < len(sector_list) and sector_list[i - 1] <= 0:
            return False
    return True


def is_trivial(a_j, trivialsectorlist):
    """
    Computes whether a particular integral is a member of a trivial sector and thus trivial.
    """
    sectornum = to_sector(a_j)
    return sectornum in trivialsectorlist


def gen_eqs(ibp, trivialsectorlist, m_vals, seeds):
    # Allow ibp to be either a filename (str) or a pre-parsed list of equation templates, for flexibility
    eq_templates = gen_eq_templates(ibp, m_vals) if isinstance(ibp, str) else ibp
    trivial_set = (
        trivialsectorlist
        if isinstance(trivialsectorlist, set)
        else set(trivialsectorlist)
    )
    d_val = m_vals["d"]
    seed_op_eq_list = []
    variables = []
    variables_set = set()
    for seed in seeds:
        tempvec = list(seed) + [d_val]
        for e_key, eq_group in eq_templates:
            equation = []
            for vec, coef_vec in eq_group:
                addvec = vadd(seed, vec)
                if not is_trivial(addvec, trivial_set):
                    addcoeff = vdot(coef_vec, tempvec)
                    if addcoeff != 0:
                        equation.append([addvec, addcoeff])
                        if addvec not in variables_set:
                            variables_set.add(addvec)
                            variables.append(addvec)
            seed_op_eq_list.append((seed, e_key, equation))

    return seed_op_eq_list, variables


def gen_all_seeds(
    top_sector: Sequence[int],
    trivialsectorlist: Sequence[int],
    s_max: int,
    r_max: int,
    d_max: int = -1,
) -> List[Tuple[int, ...]]:
    """
    Generate all seeds across all nontrivial sectors beneath top_sector.
    For each nontrivial sector, generate seeds via gen_all_seeds_in_sector,
    concatenate, then sort in reverse order with sort_integrals_desc.
    """
    sectors = enumerate_nontrivial_sectors(top_sector, trivialsectorlist)
    all_seeds: List[Tuple[int, ...]] = []
    for sec in sectors:
        r_in_sector = s_max if d_max < 0 else min(r_max, d_max + t_level(sec))
        all_seeds.extend(gen_all_seeds_in_sector(sec, s_max, r_in_sector))
    return sort_integrals_desc(all_seeds)


def gen_all_seeds_in_sector(
    sector: Sequence[int], s_max: int, r_max: int
) -> List[Tuple[int, ...]]:
    """
    Generate all seeds for a given sector.

    Constraints:
      - For each position j:
          sector[j] == 1 -> seed[j] > 0 (integer)
          sector[j] == 0 -> seed[j] <= 0 (integer)
      - Sum of positives <= r_max
      - Sum of absolute values of negatives <= s_max

    Returns:
      List of tuples of integers (same length as sector).
    """
    n = len(sector)
    ones_idx = [i for i, s in enumerate(sector) if s == 1]
    zeros_idx = [i for i, s in enumerate(sector) if s == 0]
    k1, k0 = len(ones_idx), len(zeros_idx)

    if s_max < 0 or r_max < 0:
        return []
    if k1 > 0 and r_max < k1:
        # Need at least 1 per 1-position
        return []

    # Compositions of sum_total into parts positive integers (>=1)
    def comp_pos(sum_total: int, parts: int):
        if parts == 0:
            if sum_total == 0:
                yield ()
            return
        if sum_total < parts:
            return
        if parts == 1:
            yield (sum_total,)
            return
        for first in range(1, sum_total - (parts - 1) + 1):
            for tail in comp_pos(sum_total - first, parts - 1):
                yield (first,) + tail

    # Weak compositions of sum_total into parts nonnegative integers (>=0)
    def weak_comp(sum_total: int, parts: int):
        if parts == 0:
            if sum_total == 0:
                yield ()
            return
        if parts == 1:
            yield (sum_total,)
            return
        for first in range(0, sum_total + 1):
            for tail in weak_comp(sum_total - first, parts - 1):
                yield (first,) + tail

    # Enumerate positive assignments for 1-positions
    if k1 == 0:
        ones_assignments = [()]
    else:
        ones_assignments = []
        for total_pos in range(k1, r_max + 1):
            ones_assignments.extend(comp_pos(total_pos, k1))

    # Enumerate non-positive assignments for 0-positions (store as nonnegatives then negate)
    if k0 == 0:
        zeros_assignments = [()]
    else:
        zeros_assignments = []
        for total_neg in range(0, s_max + 1):
            zeros_assignments.extend(weak_comp(total_neg, k0))

    seeds: List[Tuple[int, ...]] = []
    for o in ones_assignments:
        for z in zeros_assignments:
            seed = [0] * n
            for idx, val in zip(ones_idx, o):
                seed[idx] = val  # positive
            for idx, val in zip(zeros_idx, z):
                seed[idx] = -val  # non-positive
            seeds.append(tuple(seed))
    return seeds


def enumerate_nontrivial_sectors(
    top_sector: Sequence[int], trivialsectorlist: Sequence[int]
) -> List[Tuple[int, ...]]:
    """
    Enumerate all nontrivial sectors obtained from top_sector by optionally
    turning any 1 position into 0 (0 positions remain 0), filtered by is_trivial.
    """
    n = len(top_sector)
    ones_idx = [i for i, v in enumerate(top_sector) if v == 1]
    k = len(ones_idx)

    sectors: List[Tuple[int, ...]] = []
    for mask in range(1 << k):
        sec = [0] * n  # zeros remain 0 by construction
        for bit, idx in enumerate(ones_idx):
            sec[idx] = 1 if ((mask >> bit) & 1) else 0
        if not is_trivial(sec, trivialsectorlist):
            sectors.append(tuple(sec))
    return sectors


def spanning_cuts(
    top_sector: Sequence[int], trivialsectorlist: Sequence[int]
) -> List[Tuple[int, ...]]:
    return _spanning_cuts(enumerate_nontrivial_sectors(top_sector, trivialsectorlist))


def _spanning_cuts(sectors: List[Tuple[int, ...]]) -> List[List[int]]:
    """
    Given a list of sectors, compute a minimal set of cuts that span all the sectors.
    A cut is a tuple of integers where a positive integer at position j indicates
    that the cut requires a positive index in that position, and a non-positive integer
    indicates no requirement for that position.

    The function uses a greedy algorithm to find a small set of cuts that cover all sectors.
    """
    uncovered = set(sectors)
    cuts: List[Tuple[int, ...]] = []
    while uncovered:
        # Find the sector with the fewest positive indices (most restrictive)
        best_sector = min(uncovered, key=lambda s: sum(1 for x in s if x > 0))
        cuts.append(best_sector)
        # Remove all sectors covered by this cut (those that have positive indices at least where best_sector does)
        uncovered = {
            s
            for s in uncovered
            if not all((bs <= 0 or s[i] > 0) for i, bs in enumerate(best_sector))
        }
        cuts_as_index_positions = [_positive_index_positions(cut) for cut in cuts]
    return cuts_as_index_positions


def _positive_index_positions(sector: Tuple[int, ...]) -> List[int]:
    """Given a sector represented as a tuple of integers, return the list of
    positions where the sector has a positive integer, with one-based indexing.
    """
    return [i + 1 for i, x in enumerate(sector) if x > 0]


def r_level(a: Sequence[int]) -> int:
    """Compute the sum of positive indices in the sequence."""
    return sum(x for x in a if x > 0)


def s_level(a: Sequence[int]) -> int:
    """Compute the sum of the absolute values of negative indices in the sequence."""
    return sum(-x for x in a if x < 0)


def d_level(a: Sequence[int]) -> int:
    """Compute the total excess of propagatgor powers higher than 1 in the sequence."""
    return sum(x - 1 for x in a if x > 1)


def t_level(a: Sequence[int]) -> int:
    """Count the number of positive indices in the sequence."""
    return sum(1 for x in a if x > 0)


def compare_integrals(a: Sequence[int], b: Sequence[int]) -> int:
    """
    Comparison function for sorting integrals (tuples of indices).

    Order:
      0) by t_level (count of positive indices, smaller first),
      1) by sector number (smaller first),
      2) by sum of positive indices (smaller first),
      3) by sum of absolute values of negative indices (smaller first),
      4) lexicographic order.
    """
    ta_level = t_level(a)
    tb_level = t_level(b)
    if ta_level != tb_level:
        return -1 if ta_level < tb_level else 1

    sa = to_sector(a)
    sb = to_sector(b)
    if sa != sb:
        return -1 if sa < sb else 1

    pos_a = r_level(a)
    pos_b = r_level(b)
    if pos_a != pos_b:
        return -1 if pos_a < pos_b else 1

    neg_a = s_level(a)
    neg_b = s_level(b)
    if neg_a != neg_b:
        return -1 if neg_a < neg_b else 1

    ta, tb = tuple(a), tuple(b)
    if ta < tb:
        return -1
    if ta > tb:
        return 1
    return 0


def sort_integrals_desc(integrals: Sequence[Sequence[int]]) -> List[Tuple[int, ...]]:
    """Sort integrals in reverse order according to compare_integrals."""
    return sorted(
        (tuple(x) for x in integrals), key=cmp_to_key(compare_integrals), reverse=True
    )


def ibp_complete(
    seed_op_eq_list: list,
    seed_filter_function,
    target_integral,
    masters: list,
    modulus: int,
):
    equations = []
    variable_set = set()
    for seed, op, eq in seed_op_eq_list:
        if seed_filter_function(seed):
            equations.append(eq)
            for var, coef in eq:
                variable_set.add(var)
    variables = list(variable_set)
    return _ibp_complete(equations, variables, target_integral, set(masters), modulus)


def _ibp_complete(
    equations: list, variables: list, target_integral, master_set: set, modulus: int
):
    variables_sorted = sort_integrals_desc(variables)
    solution = solve_eqs_modulo(equations, variables_sorted, modulus)
    rhs = solution.get(target_integral, None)
    if rhs is None:
        return False
    for var, coef in rhs:
        if var not in master_set:
            return False
    return True


def run_ibp_no_reordering(
    equations: list,
    integrals: list,
    target_integral,
    masters: list,
    modulus: int,
    cost_cutoff: int = -1,
):
    reduction_complete, cost, n_eqs_used, _ = jl_feyngym.run_reduction(
        equations, integrals, [target_integral], masters, modulus, cost_cutoff
    )
    return reduction_complete, cost, n_eqs_used
