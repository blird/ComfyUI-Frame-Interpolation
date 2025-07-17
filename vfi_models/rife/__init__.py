import torch
from torch.utils.data import DataLoader
import pathlib
from vfi_utils import load_file_from_github_release, preprocess_frames, postprocess_frames, generic_frame_loop, InterpolationStateList
import typing
from comfy.model_management import get_torch_device, soft_empty_cache
import re
from functools import cmp_to_key
from packaging import version
import gc

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAME_VER_DICT = {
    "rife40.pth": "4.0",
    "rife41.pth": "4.0", 
    "rife42.pth": "4.2", 
    "rife43.pth": "4.3", 
    "rife44.pth": "4.3", 
    "rife45.pth": "4.5",
    "rife46.pth": "4.6",
    "rife47.pth": "4.7",
    "rife48.pth": "4.7",
    "rife49.pth": "4.7",
    "sudo_rife4_269.662_testV1_scale1.pth": "4.0"
    #Arch 4.10 doesn't work due to state dict mismatch
    #TODO: Investigating and fix it
    #"rife410.pth": "4.10",
    #"rife411.pth": "4.10",
    #"rife412.pth": "4.10"
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RIFE VFI": "RIFE VFI (recommend rife47 and rife49, supports batch processing)"
}

class RIFE_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    sorted(list(CKPT_NAME_VER_DICT.keys()), key=lambda ckpt_name: version.parse(CKPT_NAME_VER_DICT[ckpt_name])),
                    {"default": "rife47.pth"}
                ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 1}),
                "fast_mode": ("BOOLEAN", {"default":True}),
                "ensemble": ("BOOLEAN", {"default":True}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 32, "step": 1})
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", )
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"
    
    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames = 10,
        multiplier: typing.SupportsInt = 2,
        fast_mode = False,
        ensemble = False,
        scale_factor = 1.0,
        batch_size: typing.SupportsInt = 1,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        """
        Perform video frame interpolation using a given checkpoint model.
    
        Args:
            ckpt_name (str): The name of the checkpoint model to use.
            frames (torch.Tensor): A tensor containing input video frames.
            clear_cache_after_n_frames (int, optional): The number of frames to process before clearing CUDA cache
                to prevent memory overflow. Defaults to 10. Lower numbers are safer but mean more processing time.
                How high you should set it depends on how many input frames there are, input resolution (after upscaling),
                how many times you want to multiply them, and how long you're willing to wait for the process to complete.
            multiplier (int, optional): The multiplier for each input frame. 60 input frames * 2 = 120 output frames. Defaults to 2.
            batch_size (int, optional): Number of frame pairs to process simultaneously. Higher values = faster processing but more VRAM usage. Defaults to 1.
    
        Returns:
            tuple: A tuple containing the output interpolated frames.
    
        Note:
            This method interpolates frames in a video sequence using a specified checkpoint model. 
            It can process frames individually (batch_size=1) or in batches for significant speed improvements.
    
            To prevent memory overflow, it clears the CUDA cache after processing a specified number of frames.
        """
        from .rife_arch import IFNet
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        arch_ver = CKPT_NAME_VER_DICT[ckpt_name]
        interpolation_model = IFNet(arch_ver=arch_ver)
        interpolation_model.load_state_dict(torch.load(model_path))
        interpolation_model.eval().to(get_torch_device())
        frames = preprocess_frames(frames)
        
        # Use batch processing if batch_size > 1, otherwise use legacy sequential processing
        if batch_size > 1:
            out = self._batch_process_frames(
                frames, interpolation_model, arch_ver, multiplier, 
                fast_mode, ensemble, scale_factor, batch_size, 
                clear_cache_after_n_frames, optional_interpolation_states
            )
        else:
            def return_middle_frame(frame_0, frame_1, timestep, model, scale_list, in_fast_mode, in_ensemble):
                return model(frame_0, frame_1, timestep, scale_list, in_fast_mode, in_ensemble)
            
            scale_list = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor] 
            
            args = [interpolation_model, scale_list, fast_mode, ensemble]
            out = postprocess_frames(
                generic_frame_loop(type(self).__name__, frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                                   interpolation_states=optional_interpolation_states, dtype=torch.float32)
            )
        
        return (out,)

    def _batch_process_frames(
        self, 
        frames: torch.Tensor,
        model,
        arch_ver: str,
        multiplier: int,
        fast_mode: bool,
        ensemble: bool,
        scale_factor: float,
        batch_size: int,
        clear_cache_after_n_frames: int,
        interpolation_states: InterpolationStateList = None
    ):
        """
        Process frames in batches for improved performance.
        
        Args:
            frames: Input video frames [N, H, W, C]
            model: RIFE interpolation model
            arch_ver: Architecture version
            multiplier: Frame multiplication factor
            fast_mode: Whether to use fast mode
            ensemble: Whether to use ensemble mode
            scale_factor: Scale factor for processing
            batch_size: Number of frame pairs to process simultaneously
            clear_cache_after_n_frames: Cache clearing interval
            interpolation_states: Optional frame interpolation states
            
        Returns:
            Interpolated frames tensor
        """
        device = get_torch_device()
        dtype = torch.float32
        scale_list = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]
        
        # Create frame pairs for interpolation
        if interpolation_states is None:
            interpolation_states_list = [True] * (frames.shape[0] - 1)
        else:
            interpolation_states_list = [not interpolation_states.is_frame_skipped(i) for i in range(frames.shape[0] - 1)]
        
        enabled_frame_indices = [i for i, state in enumerate(interpolation_states_list) if state]
        
        # Dictionary to store interpolated frames by frame index and timestep
        frame_dict = {}
        
        # Add original frames to dictionary
        for i in range(frames.shape[0]):
            frame_dict[f'{i}.0'] = frames[i:i+1].to(dtype=dtype)
        
        # Process frame pairs in batches
        frames_processed = 0
        frame_indices_loader = DataLoader(enabled_frame_indices, batch_size=batch_size, shuffle=False)
        
        for batch_indices in frame_indices_loader:
            batch_indices = list(batch_indices)
            
            # Generate interpolated frames for each timestep
            for middle_i in range(1, multiplier):
                timestep = middle_i / multiplier
                
                # Prepare batch data for this timestep
                batch_frames_0 = []
                batch_frames_1 = []
                
                for frame_idx in batch_indices:
                    frame_0 = frames[frame_idx:frame_idx+1].to(dtype=torch.float32, device=device)
                    frame_1 = frames[frame_idx+1:frame_idx+2].to(dtype=torch.float32, device=device)
                    batch_frames_0.append(frame_0)
                    batch_frames_1.append(frame_1)
                
                # Stack frames for batch processing
                batch_frames_0 = torch.cat(batch_frames_0, dim=0)  # [batch_size, C, H, W]
                batch_frames_1 = torch.cat(batch_frames_1, dim=0)  # [batch_size, C, H, W]
                
                # Process entire batch at once
                with torch.no_grad():
                    batch_middle_frames = model(
                        batch_frames_0, 
                        batch_frames_1, 
                        timestep, 
                        scale_list, 
                        training=False,
                        fastmode=fast_mode, 
                        ensemble=ensemble
                    )
                
                # Store interpolated frames in dictionary
                for i, frame_idx in enumerate(batch_indices):
                    frame_dict[f'{frame_idx}.{middle_i}'] = batch_middle_frames[i:i+1].detach().cpu().to(dtype=dtype)
            
            frames_processed += len(batch_indices)
            
            # Clear cache periodically
            if frames_processed >= clear_cache_after_n_frames:
                print("Comfy-VFI: Clearing cache...", end=' ')
                soft_empty_cache()
                frames_processed = 0
                print("Done cache clearing")
            
            gc.collect()
        
        # Sort frames by key and concatenate
        sorted_keys = sorted(frame_dict.keys(), key=lambda x: (int(x.split('.')[0]), int(x.split('.')[1])))
        output_frames = torch.cat([frame_dict[key] for key in sorted_keys], dim=0)
        
        print(f"Comfy-VFI Batch Processing done! {len(output_frames)} frames generated at resolution: {output_frames[0].shape}")
        return postprocess_frames(output_frames)
