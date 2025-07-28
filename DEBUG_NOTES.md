# InverSynth Debugging Notes

## Current Status (as of commit)

### ✅ Working Components:
- **E2E CNN Model**: Fully functional, trained, and ready for evaluation
- **Dataset Generation**: 150 examples at 16kHz, 1-second samples  
- **Makefile Pipeline**: Proper dependency management and build targets
- **Audio Evaluation**: `listen_results.py` ready for model analysis

### ❌ Known Issues:

#### 1. Spectrogram CNN Models (C1, C3, C6, C6XL)
**Problem**: kapre Spectrogram layer assertion error
```
AssertionError: Hey! The input is too short!
File: /kapre/time_frequency.py, line 106
Assertion: self.len_src >= self.n_dft
```

**Root Cause**: Input format mismatch between data generator and kapre layer expectations

**What Was Tried**:
- ✅ Fixed model architecture parameter passing (`--model` instead of env var)
- ✅ Fixed input shape mismatch (data generator outputs `(samples, 1)`)
- ✅ Restored original paper STFT parameters (`n_dft=128, n_hop=64`)
- ✅ Added input length validation logic
- ❌ Still fails on kapre layer build

**Current Code State**:
- Input shape correctly set to `(16384, 1)` for channels_last
- STFT parameters: `n_dft=128, n_hop=64` (original paper values)
- Logic added to adjust parameters if input too short (but not triggered)

#### 2. Diagnosis Needed:
The kapre `Spectrogram` layer is checking `self.len_src >= self.n_dft` where:
- `len_src` seems to refer to a different dimension than expected
- Could be related to `channels_last` vs `channels_first` format
- May need different input preprocessing or kapre configuration

## Next Steps for Debugging:

1. **Investigate kapre layer expectations**:
   ```python
   # Debug what kapre actually sees
   print(f"src.shape: {src.shape}")  
   print(f"data_format: {data_format}")
   print(f"kapre input_shape: {input_shape}")
   ```

2. **Try alternative approaches**:
   - Replace kapre with `tf.signal.stft` 
   - Use different input shape format
   - Check kapre version compatibility

3. **Compare with working implementations**:
   - Check if original codebase has different kapre version
   - Look for similar projects using kapre + channels_last

## Workaround for Immediate Use:

The E2E CNN model works perfectly and can be evaluated immediately:

```bash
# Generate audio comparisons with working model
python listen_results.py

# Or run specific comparisons
make listen
make curves
```

The E2E model is actually the more advanced architecture from the paper, so full evaluation is possible without the spectrogram models.

## Files Modified in This Debug Session:

- `models/spectrogram_cnn.py`: Input shape fixes, STFT parameter restoration
- `Makefile`: Corrected model parameter passing 
- `doit`: Updated to use `--model` flag consistently
- `.gitignore`: Added debug output files

## Test Commands:

```bash
# Working:
make model-e2e     # ✅ Already trained
python listen_results.py  # ✅ Should work

# Broken:
make model-c1      # ❌ kapre assertion error
make model-c3      # ❌ Same issue expected
make model-c6      # ❌ Same issue expected  
make model-c6xl    # ❌ Same issue expected
```