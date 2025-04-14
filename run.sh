# Add the following line to enable test mode; otherwise, it defaults to validation mode
export TEST_MODE=True

# Baseline: greedy decoding
python3 main_aro.py --dataset=Controlled_Images_A --model-name='llava1.5' --download --method=base  --option=four

# For Scaling_Vis on Controlled_A, a weight of 0.8 is used
python3 main_aro.py --dataset=Controlled_Images_A --model-name='llava1.5' --download --method=scaling_vis  --weight=0.8  --option=four

# For Adapt_Vis on Controlled_A, weight1 is set to 0.5, weight2 to 1.5, and threshold to 0.4
python3 main_aro.py --dataset=Controlled_Images_A --model-name='llava1.5' --download --method adapt_vis --weight1 0.5  --weight2 1.5 --threshold 0.4 --option=four
