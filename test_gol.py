import pytest
import torch
import numpy as np

# Import all the gol functions
from gol_torch import (
    gol_torch_conv2d, gol_torch_conv2d_compiled, 
    gol_torch_conv2d_f16, gol_torch_conv2d_f16_compiled,
    gol_torch_sum, gol_torch_sum_compiled
)
from gol_cuda import (
    gol_cuda, gol_cuda_shared_memory, gol_cuda_wideload, 
    gol_cuda_grouped, gol_cuda_bitpacked, gol_cuda_bitpacked_64, 
    gol_cuda_grouped_bitpacked_64, gol_cuda_grouped_bitpacked_64_multistep
)
from gol_triton import (
    gol_triton_1d, gol_triton_2d, gol_triton_8bit_1d, 
    gol_triton_32bit_1d, gol_triton_64bit_1d
)
from utils import bit_encode, bit_decode, long_encode, long_decode, longlong_encode, longlong_decode

device = torch.device('cuda:0')

class TestGameOfLife:
    """Test suite for Game of Life implementations using gol_torch_sum as reference."""
    
    @pytest.fixture
    def reference_implementation(self):
        """Use gol_torch_sum as the reference implementation."""
        return gol_torch_sum
    
    def run_reference(self, pattern, steps=1):
        """Run the reference implementation for comparison."""
        for _ in range(steps):
            pattern = gol_torch_sum(pattern)
        return pattern
    
    def compare_interiors(self, result, reference, original=None):
        """Compare only the interior cells, ignoring boundaries."""
        # Many of our kernels behave badly on the boundaries, so we need to ignore them
        inset = 64
        result_interior = result[inset:-inset, inset:-inset]
        reference_interior = reference[inset:-inset, inset:-inset]
        
        if not torch.equal(result_interior, reference_interior):
            # Find differences
            diff_mask = result_interior != reference_interior
            diff_positions = torch.nonzero(diff_mask, as_tuple=False)
            
            if len(diff_positions) > 0:
                # Show first few differences
                num_diffs_to_show = min(10, len(diff_positions))
                diff_details = []
                
                # For the first difference, show 3x3 context
                first_pos = diff_positions[0]
                first_row, first_col = first_pos[0].item(), first_pos[1].item()
                
                # Get 3x3 context (need to account for boundary offset)
                orig_row, orig_col = first_row + inset, first_col + inset  # +1 because we removed boundary
                expected_3x3 = reference[orig_row-1:orig_row+2, orig_col-1:orig_col+2].cpu().numpy()
                actual_3x3 = result[orig_row-1:orig_row+2, orig_col-1:orig_col+2].cpu().numpy()
                
                diff_details.append(f"  First difference at position (0x{orig_row:x}, 0x{orig_col:x}):")
                if original is not None:
                    before_3x3 = original[orig_row-1:orig_row+2, orig_col-1:orig_col+2].cpu().numpy()
                    diff_details.append(f"    Before (3x3):\n{np.array2string(before_3x3, separator=', ')}")
                diff_details.append(f"    Expected (3x3):\n{np.array2string(expected_3x3, separator=', ')}")
                diff_details.append(f"    Actual (3x3):\n{np.array2string(actual_3x3, separator=', ')}")
                
                # Show remaining differences
                for i in range(1, num_diffs_to_show):
                    pos = diff_positions[i]
                    row, col = pos[0].item(), pos[1].item()
                    orig_row, orig_col = row + inset, col + inset  # +1 because we removed boundary
                    result_val = result_interior[row, col].item()
                    ref_val = reference_interior[row, col].item()
                    diff_details.append(f"  Position (0x{orig_row:x}, 0x{orig_col:x}): got {result_val}, expected {ref_val}")
                
                diff_summary = f"\nFound {len(diff_positions)} differences in interior cells:\n" + "\n".join(diff_details)
                if len(diff_positions) > num_diffs_to_show:
                    diff_summary += f"\n  ... and {len(diff_positions) - num_diffs_to_show} more differences"
                
                raise AssertionError(diff_summary)
        
        return True
    
    def test_gol_torch_conv2d(self, reference_implementation):
        """Test gol_torch_conv2d function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern)
        result = gol_torch_conv2d(pattern)
        
        assert self.compare_interiors(result, reference_result, pattern), "gol_torch_conv2d failed"
    
    def test_gol_torch_conv2d_compiled(self, reference_implementation):
        """Test gol_torch_conv2d_compiled function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern)
        result = gol_torch_conv2d_compiled(pattern)
        
        assert self.compare_interiors(result, reference_result, pattern), "gol_torch_conv2d_compiled failed"
    
    def test_gol_torch_conv2d_f16(self, reference_implementation):
        """Test gol_torch_conv2d_f16 function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.float16)
        
        reference_result = self.run_reference(pattern.to(torch.int8))
        result = gol_torch_conv2d_f16(pattern)
        
        assert self.compare_interiors(result.to(torch.int8), reference_result, pattern.to(torch.int8)), "gol_torch_conv2d_f16 failed"
    
    def test_gol_torch_conv2d_f16_compiled(self, reference_implementation):
        """Test gol_torch_conv2d_f16_compiled function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.float16)
        
        reference_result = self.run_reference(pattern.to(torch.int8))
        result = gol_torch_conv2d_f16_compiled(pattern)
        
        assert self.compare_interiors(result.to(torch.int8), reference_result, pattern.to(torch.int8)), "gol_torch_conv2d_f16_compiled failed"
    
    def test_gol_torch_sum(self, reference_implementation):
        """Test gol_torch_sum function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern)
        result = gol_torch_sum(pattern)
        
        assert self.compare_interiors(result, reference_result, pattern), "gol_torch_sum failed"
    
    def test_gol_torch_sum_compiled(self, reference_implementation):
        """Test gol_torch_sum_compiled function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern)
        result = gol_torch_sum_compiled(pattern)
        
        assert self.compare_interiors(result, reference_result, pattern), "gol_torch_sum_compiled failed"
    
    def test_gol_cuda(self, reference_implementation):
        """Test gol_cuda function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern)
        result = gol_cuda(pattern)
        
        assert self.compare_interiors(result, reference_result, pattern), "gol_cuda failed"
    
    def test_gol_cuda_shared_memory(self, reference_implementation):
        """Test gol_cuda_shared_memory function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern)
        result = gol_cuda_shared_memory(pattern)
        
        assert self.compare_interiors(result, reference_result, pattern), "gol_cuda_shared_memory failed"
    
    def test_gol_cuda_wideload(self, reference_implementation):
        """Test gol_cuda_wideload function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern)
        result = gol_cuda_wideload(pattern)
        
        assert self.compare_interiors(result, reference_result, pattern), "gol_cuda_wideload failed"
    
    def test_gol_cuda_grouped(self, reference_implementation):
        """Test gol_cuda_grouped function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern)
        result = gol_cuda_grouped(pattern)
        
        assert self.compare_interiors(result, reference_result, pattern), "gol_cuda_grouped failed"
    
    def test_gol_cuda_bitpacked(self, reference_implementation):
        """Test gol_cuda_bitpacked function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern)
        encoded_pattern = bit_encode(pattern)
        result = gol_cuda_bitpacked(encoded_pattern)
        decoded_result = bit_decode(result)
        
        assert self.compare_interiors(decoded_result, reference_result, pattern), "gol_cuda_bitpacked failed"
    
    def test_gol_cuda_bitpacked_64(self, reference_implementation):
        """Test gol_cuda_bitpacked_64 function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern)
        encoded_pattern = longlong_encode(pattern)
        result = gol_cuda_bitpacked_64(encoded_pattern)
        decoded_result = longlong_decode(result)
        
        assert self.compare_interiors(decoded_result, reference_result, pattern), "gol_cuda_bitpacked_64 failed"
    
    def test_gol_cuda_grouped_bitpacked_64(self, reference_implementation):
        """Test gol_cuda_grouped_bitpacked_64 function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern)
        encoded_pattern = longlong_encode(pattern)
        result = gol_cuda_grouped_bitpacked_64(encoded_pattern)
        decoded_result = longlong_decode(result)
        
        assert self.compare_interiors(decoded_result, reference_result, pattern), "gol_cuda_grouped_bitpacked_64 failed"
    
    def test_gol_cuda_grouped_bitpacked_64_multistep(self, reference_implementation):
        """Test gol_cuda_grouped_bitpacked_64_multistep function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern, steps=4)
        encoded_pattern = longlong_encode(pattern)
        result = gol_cuda_grouped_bitpacked_64_multistep(encoded_pattern)
        decoded_result = longlong_decode(result)
        
        assert self.compare_interiors(decoded_result, reference_result, pattern), "gol_cuda_grouped_bitpacked_64_multistep failed"
    
    def test_gol_triton_1d(self, reference_implementation):
        """Test gol_triton_1d function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern)
        result = gol_triton_1d(pattern)
        
        assert self.compare_interiors(result, reference_result, pattern), "gol_triton_1d failed"
    
    def test_gol_triton_2d(self, reference_implementation):
        """Test gol_triton_2d function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern)
        result = gol_triton_2d(pattern)
        
        assert self.compare_interiors(result, reference_result, pattern), "gol_triton_2d failed"
    
    def test_gol_triton_8bit_1d(self, reference_implementation):
        """Test gol_triton_8bit_1d function."""
        torch.manual_seed(44)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern)
        encoded_pattern = bit_encode(pattern)
        result = gol_triton_8bit_1d(encoded_pattern)
        decoded_result = bit_decode(result)
        
        assert self.compare_interiors(decoded_result, reference_result, pattern), "gol_triton_8bit_1d failed"
    
    def test_gol_triton_32bit_1d(self, reference_implementation):
        """Test gol_triton_32bit_1d function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern)
        encoded_pattern = long_encode(pattern)
        result = gol_triton_32bit_1d(encoded_pattern)
        decoded_result = long_decode(result)
        
        assert self.compare_interiors(decoded_result, reference_result, pattern), "gol_triton_32bit_1d failed"
    
    def test_gol_triton_64bit_1d(self, reference_implementation):
        """Test gol_triton_64bit_1d function."""
        torch.manual_seed(42)
        pattern = (torch.rand(2048, 2048, device=device) < 0.3).to(torch.int8)
        
        reference_result = self.run_reference(pattern)
        encoded_pattern = longlong_encode(pattern)
        result = gol_triton_64bit_1d(encoded_pattern)
        decoded_result = longlong_decode(result)
        
        assert self.compare_interiors(decoded_result, reference_result, pattern), "gol_triton_64bit_1d failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])