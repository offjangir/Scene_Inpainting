"""
Quick test script for inference pipeline.

This script helps verify that your trained model can be loaded and used for inference.
"""

import os
import sys
from inference import InpaintingInference

def test_model_loading(controlnet_path):
    """Test that the model can be loaded successfully."""
    print("="*70)
    print("Testing Model Loading")
    print("="*70)
    
    try:
        inferencer = InpaintingInference(
            controlnet_path=controlnet_path,
            device="cuda"
        )
        print("✅ Model loaded successfully!")
        return inferencer
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_single_inference(inferencer, mask_path, cond_path, output_path):
    """Test single image inference."""
    print("\n" + "="*70)
    print("Testing Single Image Inference")
    print("="*70)
    
    if not os.path.exists(mask_path):
        print(f"❌ Mask not found: {mask_path}")
        return False
    
    if not os.path.exists(cond_path):
        print(f"❌ Conditioning image not found: {cond_path}")
        return False
    
    try:
        result = inferencer.inpaint_single(
            mask_path=mask_path,
            cond_path=cond_path,
            output_path=output_path,
            num_inference_steps=20,  # Quick test with fewer steps
            guidance_scale=7.5,
            seed=42
        )
        print("✅ Single inference test passed!")
        return True
    except Exception as e:
        print(f"❌ Single inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_inference(inferencer, input_dir, output_dir):
    """Test batch inference."""
    print("\n" + "="*70)
    print("Testing Batch Inference")
    print("="*70)
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    try:
        inferencer.inpaint_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            num_inference_steps=20,  # Quick test with fewer steps
            guidance_scale=7.5,
            seed=42
        )
        print("✅ Batch inference test passed!")
        return True
    except Exception as e:
        print(f"❌ Batch inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test inference pipeline")
    parser.add_argument("--controlnet_path", type=str, required=True,
                       help="Path to trained ControlNet")
    parser.add_argument("--mask_path", type=str,
                       help="Path to test mask (for single inference test)")
    parser.add_argument("--cond_path", type=str,
                       help="Path to test conditioning image (for single inference test)")
    parser.add_argument("--input_dir", type=str,
                       help="Path to test directory (for batch inference test)")
    parser.add_argument("--skip_single", action="store_true",
                       help="Skip single inference test")
    parser.add_argument("--skip_batch", action="store_true",
                       help="Skip batch inference test")
    
    args = parser.parse_args()
    
    # Test 1: Load model
    inferencer = test_model_loading(args.controlnet_path)
    if inferencer is None:
        print("\n❌ Model loading failed. Cannot proceed with inference tests.")
        sys.exit(1)
    
    # Test 2: Single inference (if paths provided)
    single_test_passed = True
    if not args.skip_single and args.mask_path and args.cond_path:
        output_path = "test_output_single.png"
        single_test_passed = test_single_inference(
            inferencer, args.mask_path, args.cond_path, output_path
        )
    elif not args.skip_single:
        print("\n⚠️  Skipping single inference test (no --mask_path or --cond_path provided)")
    
    # Test 3: Batch inference (if input_dir provided)
    batch_test_passed = True
    if not args.skip_batch and args.input_dir:
        output_dir = "test_output_batch"
        batch_test_passed = test_batch_inference(
            inferencer, args.input_dir, output_dir
        )
    elif not args.skip_batch:
        print("\n⚠️  Skipping batch inference test (no --input_dir provided)")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Model Loading:      ✅ PASSED")
    if not args.skip_single and args.mask_path and args.cond_path:
        print(f"Single Inference:   {'✅ PASSED' if single_test_passed else '❌ FAILED'}")
    if not args.skip_batch and args.input_dir:
        print(f"Batch Inference:    {'✅ PASSED' if batch_test_passed else '❌ FAILED'}")
    print("="*70)
    
    if single_test_passed and batch_test_passed:
        print("\n🎉 All tests passed! Your inference pipeline is ready to use.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

