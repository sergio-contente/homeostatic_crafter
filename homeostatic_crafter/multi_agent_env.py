import collections
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import gymnasium as gym

from . import constants
from . import engine
from . import objects
from . import worldgen

# Attempt to import pygame for rendering, but make it optional
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

class MultiAgentEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30,
        'name': "MultiAgentHomeostaticCrafter-v0"
    }

    def __init__(self, 
                 num_agents: int = 2, 
                 area: Tuple[int, int] = (64, 64), 
                 view: Tuple[int, int] = (9, 9), 
                 size: Tuple[int, int] = (64, 64), # Output image size (W, H) for observations
                 reward: bool = True, 
                 length: int = 10000, 
                 seed: Optional[int] = None,
                 random_internal: bool = False,
                 render_mode: Optional[str] = None,
                 homeostatic: bool = True):
        super().__init__()

        if num_agents <= 0:
            raise ValueError("Number of agents must be positive.")
        self.num_agents = num_agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agent_id_to_index = {agent_id: i for i, agent_id in enumerate(self.possible_agents)}

        self._area = area
        self._view_grid_size = np.array(view if hasattr(view, '__len__') else (view, view)) # Grid units (e.g. 9x9 tiles)
        self._obs_image_size = np.array(size if hasattr(size, '__len__') else (size, size)) # Pixel size (W, H) for obs image
        self._reward_enabled = reward
        self._length = length
        self._seed = np.random.randint(0, 2**31 - 1) if seed is None else seed
        self._episode = 0
        self._homeostatic_rewards = homeostatic
        self._random_internal_player_state = random_internal
        
        self.render_mode = render_mode
        self._pygame_module = None
        self.screen = None
        self.clock = None

        if self.render_mode == "human":
            if not PYGAME_AVAILABLE:
                print("Warning: Pygame is not installed. Human rendering mode is not available. Falling back to rgb_array.")
                self.render_mode = "rgb_array"
            else:
                self._pygame_module = pygame
                self._initialize_pygame()

        self._world = engine.World(self._area, constants.materials, (12, 12)) # Chunk size fixed
        self._textures = engine.Textures(constants.root / 'assets')

        # Calculate grid dimensions
        total_grid_height = self._view_grid_size[1]
        item_rows = int(np.ceil(len(constants.items) / self._view_grid_size[0]))
        local_view_height = total_grid_height - item_rows
        
        # Ensure both views use the same width and appropriate heights
        self._local_view_obj = engine.LocalView(
            self._world, self._textures, [self._view_grid_size[0], local_view_height]
        )
        self._item_view_obj = engine.ItemView(
            self._textures, [self._view_grid_size[0], item_rows]
        )
        self._sem_view_obj = engine.SemanticView(self._world, [
            objects.Player, objects.Cow, objects.Zombie,
            objects.Skeleton, objects.Arrow, objects.Plant
        ])

        self._step_count: Optional[int] = None
        self._players: List[objects.Player] = []
        self._last_interos: Dict[str, np.ndarray] = {}
        self._last_healths: Dict[str, float] = {}
        self._unlocked_achievements: set[str] = set()

        self._intero_normalizer = np.array([
            constants.items['health']['max'],
            constants.items['food']['max'],
            constants.items['drink']['max'],
            constants.items['energy']['max']
        ], dtype=np.float32)
        
        # Define action and observation spaces
        # Observation: visual (Box) + measurements (Box)
        single_agent_obs_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(0, 255, (3, self._obs_image_size[0], self._obs_image_size[1]), dtype=np.uint8), # C, W, H
            "measurements": gym.spaces.Box(0, 1, (4,), dtype=np.float32),
        })
        single_agent_action_space = gym.spaces.Discrete(len(constants.actions))

        self.observation_space = gym.spaces.Dict(
            {agent_id: single_agent_obs_space for agent_id in self.possible_agents}
        )
        self.action_space = gym.spaces.Dict(
            {agent_id: single_agent_action_space for agent_id in self.possible_agents}
        )
        self.agents = [] # Will be populated in reset()

    def _initialize_pygame(self):
        if self._pygame_module and self.render_mode == "human":
            self._pygame_module.init()
            # Render size for screen should match observation image size for simplicity, or be configurable
            pygame_screen_size = self._obs_image_size 
            self.screen = self._pygame_module.display.set_mode(pygame_screen_size)
            self._pygame_module.display.set_caption(f"Multi-Agent Homeostatic Crafter (View: {self.possible_agents[0]})")
            self.clock = self._pygame_module.time.Clock()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self._seed = seed
        
        self.agents = list(self.possible_agents) # All agents are active from the start
        self._episode += 1
        self._step_count = 0
        
        current_seed = hash((self._seed, self._episode)) % (2**31 - 1)
        self._world.reset(seed=current_seed)
        self._update_time()

        self._players = []
        self._last_interos.clear()
        self._last_healths.clear()
        self._unlocked_achievements.clear()

        center_x, center_y = self._world.area[0] // 2, self._world.area[1] // 2
        for i, agent_id in enumerate(self.agents):
            # Simple offset spawning, ensure positions are within bounds and valid if possible
            # Player constructor handles initial position adjustment if on invalid tile
            angle = 2 * np.pi * i / self.num_agents
            offset_x = int(3 * np.cos(angle)) # Spread them out a bit
            offset_y = int(3 * np.sin(angle))
            start_pos = (center_x + offset_x, center_y + offset_y)
            
            player = objects.Player(self._world, start_pos, 
                                    random_internal=self._random_internal_player_state, 
                                    name=agent_id) # Name player object for clarity
            self._players.append(player)
            self._world.add(player)
            self._last_interos[agent_id] = player.get_interoception()
            self._last_healths[agent_id] = player.health
        
        if self._players:
            worldgen.generate_world(self._world, self._players[0]) # Use first player for reference
        else:
            worldgen.generate_world(self._world, None)

        observations = {agent_id: self._get_obs_for_player(self._players[i]) for i, agent_id in enumerate(self.agents)}
        infos = {agent_id: self._get_info_for_player(self._players[i]) for i, agent_id in enumerate(self.agents)}
        
        return observations, infos

    def _get_obs_for_player(self, player: objects.Player) -> Dict[str, np.ndarray]:
        # Image is C, W, H for observation space
        vision_chw = self._get_image_for_player(player, size=self._obs_image_size)
        norm_intero = player.get_interoception() / self._intero_normalizer
        return {"obs": vision_chw, "measurements": norm_intero}

    def _get_image_for_player(self, player: objects.Player, size: Tuple[int, int]) -> np.ndarray:
        # size is (W_target, H_target) for the output image
        # Calculate unit size based on the grid width only to maintain aspect ratio
        unit_size = int(size[0] / self._view_grid_size[0])
        unit = np.array([unit_size, unit_size], dtype=np.int32)

        # Get views with consistent unit size
        local_img_hwc = self._local_view_obj(player, unit)
        item_img_hwc = self._item_view_obj(player.inventory, unit)
        
        # Ensure both images have the same width
        if local_img_hwc.shape[1] != item_img_hwc.shape[1]:
            target_width = max(local_img_hwc.shape[1], item_img_hwc.shape[1])
            # Pad the narrower image
            if local_img_hwc.shape[1] < target_width:
                pad_width = target_width - local_img_hwc.shape[1]
                local_img_hwc = np.pad(
                    local_img_hwc,
                    ((0, 0), (0, pad_width), (0, 0)),
                    mode='constant'
                )
            if item_img_hwc.shape[1] < target_width:
                pad_width = target_width - item_img_hwc.shape[1]
                item_img_hwc = np.pad(
                    item_img_hwc,
                    ((0, 0), (0, pad_width), (0, 0)),
                    mode='constant'
                )
        
        # Concatenate vertically (axis 0 for HWC format)
        full_view_hwc = np.concatenate([local_img_hwc, item_img_hwc], axis=0)
        
        # Create output canvas with target size
        canvas_h, canvas_w = size[1], size[0]
        canvas_hwc = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Calculate centering offsets
        start_x = max(0, (canvas_w - full_view_hwc.shape[1]) // 2)
        start_y = max(0, (canvas_h - full_view_hwc.shape[0]) // 2)
        
        # Calculate the region to copy
        copy_h = min(full_view_hwc.shape[0], canvas_h - start_y)
        copy_w = min(full_view_hwc.shape[1], canvas_w - start_x)
        
        if copy_h > 0 and copy_w > 0:
            canvas_hwc[start_y:start_y + copy_h, start_x:start_x + copy_w] = \
                full_view_hwc[:copy_h, :copy_w]
        
        # Convert HWC to CHW for observation space (C, W, H)
        return canvas_hwc.transpose((2, 1, 0))
    
    def _get_reward_for_player(self, player: objects.Player, agent_id: str) -> float:
        norm_intero = player.get_interoception() / self._intero_normalizer
        last_norm_intero = self._last_interos[agent_id] / self._intero_normalizer
        
        def drive_fn(x_intero):
            r_drive = 0
            for i, val_name in enumerate(['health', 'food', 'drink', 'energy']):
                r_drive += constants.homeostasis['scale'][val_name] * \
                           (x_intero[i] - constants.homeostasis['target'][val_name]) ** 2
            return r_drive
        
        current_drive = drive_fn(norm_intero)
        last_drive = drive_fn(last_norm_intero)
        homeostatic_reward = last_drive - current_drive
        return homeostatic_reward

    def _get_info_for_player(self, player: objects.Player) -> Dict[str, Any]:
        # Basic info, can be expanded
        return {
            'inventory': player.inventory.copy(),
            'achievements': player.achievements.copy(), # Per-player achievements
            'player_pos': player.pos,
            'interoception': player.get_interoception(),
            'health': player.health,
            'is_sleeping': player.sleeping,
        }

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        self._step_count += 1
        self._update_time()

        observations = {}
        rewards = {}
        terminations = {} # Agent is "terminated" (e.g. dead)
        truncations = {}  # Agent is "truncated" (e.g. episode length)
        infos = {}

        active_agents_after_step = []

        for i, agent_id in enumerate(self.agents): # Iterate over currently active agents
            player = self._players[self.agent_id_to_index[agent_id]] # Get player object
            action_val = actions.get(agent_id)

            if action_val is None: # Should not happen if controller sends actions for all active agents
                player.action = 'noop'
            else:
                player.action = constants.actions[action_val]
        
        # Update all world objects (including players executing their set actions)
        for obj in self._world.objects:
            is_near_any_player = False
            for p_obj in self._players: # Check against all player objects
                if p_obj.health > 0 and p_obj.distance(obj) < 2 * max(self._view_grid_size):
                    is_near_any_player = True
                    break
            if is_near_any_player or isinstance(obj, objects.Player):
                obj.update()
        
        # Creature balancing (referencing all players)
        if self._step_count % 10 == 0:
            for chunk_pos, objs_in_chunk in self._world.chunks.items():
                self._balance_chunk_multi_agent(chunk_pos, objs_in_chunk, self._players)

        # Collect results for each agent
        for i, agent_id in enumerate(self.agents):
            player = self._players[self.agent_id_to_index[agent_id]]
            player.update_interoception() # Update internal state like hunger, thirst

            observations[agent_id] = self._get_obs_for_player(player)
            
            current_reward = 0.0
            if self._homeostatic_rewards:
                current_reward += self._get_reward_for_player(player, agent_id)
            else: # Original Crafter reward logic (simplified for MA)
                current_reward += (player.health - self._last_healths[agent_id]) / 10.0

            self._last_interos[agent_id] = player.get_interoception()
            self._last_healths[agent_id] = player.health

            # Achievement reward (per player, unlocks are shared for global tracking)
            # The reward itself is individual if not homeostatic
            player_unlocked_now = {
                name for name, count in player.achievements.items()
                if count > 0 and name not in self._unlocked_achievements
            }
            if player_unlocked_now:
                self._unlocked_achievements.update(player_unlocked_now)
                if not self._homeostatic_rewards:
                    current_reward += 1.0 * len(player_unlocked_now) 
            
            rewards[agent_id] = current_reward if self._reward_enabled else 0.0
            
            terminated = player.health <= 0
            truncated = self._length and self._step_count >= self._length
            
            terminations[agent_id] = terminated
            truncations[agent_id] = truncated
            infos[agent_id] = self._get_info_for_player(player)
            infos[agent_id]['discount'] = 1.0 - float(terminated) # For RL

            if not (terminated or truncated):
                active_agents_after_step.append(agent_id)
        
        self.agents = active_agents_after_step # Update list of active agents

        # If all agents are done, the episode is effectively over for the environment
        if not self.agents:
            # This is like a global done. Individual dicts will reflect agent status.
            pass

        return observations, rewards, terminations, truncations, infos

    def _update_time(self):
        progress = (self._step_count / 300) % 1 + 0.3 # Matches original
        daylight = 1 - np.abs(np.cos(np.pi * progress)) ** 3
        self._world.daylight = daylight

    def _balance_chunk_multi_agent(self, chunk_pos: Tuple[int, int], objs_in_chunk: List[object], all_players: List[objects.Player]):
        # Reference player for mob targeting (simplification)
        ref_player = all_players[0] if all_players else None 
        if not ref_player: return

        light = self._world.daylight
        self._balance_object_multi_agent(
            chunk_pos, objs_in_chunk, objects.Zombie, 'grass', 6, 0, 0.3, 0.4,
            lambda pos: objects.Zombie(self._world, pos, ref_player),
            lambda num, space: (0 if space < 50 else 3.5 - 3 * light, 3.5 - 3 * light),
            all_players)
        self._balance_object_multi_agent(
            chunk_pos, objs_in_chunk, objects.Skeleton, 'path', 7, 7, 0.1, 0.1,
            lambda pos: objects.Skeleton(self._world, pos, ref_player),
            lambda num, space: (0 if space < 6 else 1, 2),
            all_players)
        self._balance_object_multi_agent(
            chunk_pos, objs_in_chunk, objects.Cow, 'grass', 5, 5, 0.01, 0.1,
            lambda pos: objects.Cow(self._world, pos),
            lambda num, space: (0 if space < 30 else 1, 1.5 + light),
            all_players)

    def _balance_object_multi_agent(
            self, chunk_coords_key: Tuple[int, int], objs_in_chunk: List[object], 
            obj_class: type, material: str, 
            span_dist: float, despan_dist: float, spawn_prob: float, despawn_prob: float, 
            constructor: callable, target_fn: callable, all_players: List[objects.Player]):
        
        # chunk_coords_key is already the tuple (xmin, xmax, ymin, ymax)
        xmin, xmax, ymin, ymax = chunk_coords_key
        random_gen = self._world.random
        creatures = [obj for obj in objs_in_chunk if isinstance(obj, obj_class)]
        
        # Mask for valid spawn locations (on specified material)
        mask = self._world.mask(xmin, xmax, ymin, ymax, material)
        
        target_min, target_max = target_fn(len(creatures), mask.sum())

        if len(creatures) < int(target_min) and random_gen.uniform() < spawn_prob:
            valid_indices_y, valid_indices_x = np.where(mask)
            if not valid_indices_y.size: return # No valid spots

            chosen_idx = random_gen.randint(0, len(valid_indices_y))
            # valid_indices are relative to the chunk's slice, convert to world coords
            pos_y_chunk_relative, pos_x_chunk_relative = valid_indices_y[chosen_idx], valid_indices_x[chosen_idx]
            pos = np.array((xmin + pos_x_chunk_relative, ymin + pos_y_chunk_relative))

            empty = self._world[pos][1] is None
            away_from_all = True
            if empty:
                for player in all_players:
                    if player.health > 0 and player.distance(pos) < span_dist:
                        away_from_all = False
                        break
                if away_from_all:
                    self._world.add(constructor(pos))

        elif len(creatures) > int(target_max) and random_gen.uniform() < despawn_prob:
            if not creatures: return
            obj_to_despawn = creatures[random_gen.randint(0, len(creatures))]
            
            far_from_all = True
            for player in all_players:
                if player.health > 0 and player.distance(obj_to_despawn.pos) < despan_dist:
                    far_from_all = False # Too close to a player, don't despawn
                    break
            if far_from_all:
                self._world.remove(obj_to_despawn)

    def render(self, agent_id_to_render: Optional[str] = None) -> Optional[np.ndarray]:
        if self.render_mode not in ['human', 'rgb_array']:
            # Fallback for unsupported modes or if called unexpectedly
            if PYGAME_AVAILABLE and self.render_mode == 'human' and self._pygame_module and self.screen:
                 # If human mode was intended but failed, try to show black screen
                self.screen.fill((0,0,0))
                self._pygame_module.display.flip()
                if self.clock: self.clock.tick(self.metadata["render_fps"])
            return None
        
        if not self.agents: # No active agents to render
             if self.render_mode == "human" and self._pygame_module and self.screen :
                self.screen.fill((0,0,0)) # Black screen
                self._pygame_module.display.flip()
                if self.clock: self.clock.tick(self.metadata["render_fps"])
             elif self.render_mode == "rgb_array":
                return np.zeros((self._obs_image_size[1], self._obs_image_size[0], 3), dtype=np.uint8) # HWC
             return None

        render_agent_id = agent_id_to_render if agent_id_to_render in self.agents else self.agents[0]
        player_idx = self.agent_id_to_index[render_agent_id]
        player_to_render = self._players[player_idx]

        # _get_image_for_player returns C, W, H (matching obs_space format)
        img_chw = self._get_image_for_player(player_to_render, size=self._obs_image_size)
        
        if self.render_mode == "human":
            if not self._pygame_module or not self.screen: return None
            
            img_whc_for_pygame = img_chw.transpose((1, 2, 0)) # CWH -> WHC
            surface = self._pygame_module.surfarray.make_surface(img_whc_for_pygame)
            self.screen.blit(surface, (0, 0))
            self._pygame_module.event.pump()
            self._pygame_module.display.set_caption(f"Multi-Agent Crafter (View: {render_agent_id}, Step: {self._step_count})")
            self._pygame_module.display.flip()
            if self.clock: self.clock.tick(self.metadata["render_fps"])
            return None
        
        elif self.render_mode == "rgb_array":
            img_hwc_for_return = img_chw.transpose((2, 1, 0)) # CWH -> HWC
            return img_hwc_for_return
        
        return None # Should be covered by initial check

    def close(self):
        if self.render_mode == "human" and self._pygame_module:
            self._pygame_module.display.quit()
            self._pygame_module.quit()
            self.screen = None
            self.clock = None
            self._pygame_module = None # Ensure it's cleared

    @property
    def action_names(self) -> List[str]: # Global action names
        return constants.actions

# Example of how to use (would be in a separate main script)
if __name__ == '__main__':
    print("Creating Multi-Agent Crafter Environment (example usage)")
    num_ma_agents = 2
    # Test with human rendering if available, otherwise rgb_array
    preferred_render_mode = 'human' if PYGAME_AVAILABLE else 'rgb_array'
    print(f"Attempting render_mode: {preferred_render_mode}")

    env = MultiAgentEnv(num_agents=num_ma_agents, render_mode=preferred_render_mode, size=(256,256)) 
    
    print(f"Number of agents: {env.num_agents}")
    print(f"Possible agents: {env.possible_agents}")
    # print(f"Observation Space for agent_0: {env.observation_space['agent_0']}")
    # print(f"Action Space for agent_0: {env.action_space['agent_0']}")

    for ep in range(1):
        print(f"\n--- Episode {ep + 1} ---")
        observations, infos = env.reset()
        # print(f"Initial obs (agent_0 shape): {observations['agent_0']['obs'].shape}")
        # print(f"Initial measurements (agent_0): {observations['agent_0']['measurements']}")
        
        episode_rewards = {agent_id: 0.0 for agent_id in env.agents}
        
        for step in range(150): # Short run
            if not env.agents: # All agents are done
                print(f"All agents done at step {step}. Episode terminated early.")
                break

            # Generate random actions for all active agents
            actions_to_take = {
                agent_id: env.action_space[agent_id].sample() 
                for agent_id in env.agents
            }
            
            # print(f"Step {step + 1}, Active Agents: {env.agents}, Actions: { {k: constants.actions[v] for k,v in actions_to_take.items()} }")
            
            next_observations, rewards, terminations, truncations, infos = env.step(actions_to_take)
            
            for agent_id in rewards: 
                 episode_rewards[agent_id] += rewards[agent_id]
            
            if preferred_render_mode == 'human':
               env.render(agent_id_to_render="agent_1") 
            elif preferred_render_mode == 'rgb_array':
               img_array = env.render() 
               if step % 50 == 0: # Print shape occasionally
                   pass # print(f"  RGB array shape: {img_array.shape if img_array is not None else 'None'}")

            # Update observations for next step
            observations = next_observations
            
            if not env.agents:
                 print(f"All agents are done at step {step + 1} (after processing step results).")
                 break
        
        print(f"Episode {ep+1} finished. Total rewards: {episode_rewards}")

    env.close()
    print("\nMulti-Agent Env Test Complete.") 
