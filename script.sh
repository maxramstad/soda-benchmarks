#! /bin/bash

# python workflow.py --kernel bicg --dimension MINI --target Transformed 
# python workflow.py --kernel atax --dimension MINI --target Transformed 
# python workflow.py --kernel gemm --dimension MINI --target Transformed 
# python workflow.py --kernel gemver --dimension MINI --target Transformed 
# python workflow.py --kernel gesummv --dimension MINI --target Transformed 
python workflow.py --kernel symm --dimension MINI --target Transformed 
# python workflow.py --kernel syr2k --dimension MINI --target Transformed 
# python workflow.py --kernel syrk --dimension MINI --target Transformed 
python workflow.py --kernel trmm --dimension MINI --target Transformed 
