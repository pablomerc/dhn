##################################################
# AutoRegression
##################################################

EXP_NAME=sinpend_kernel2_stride1
#EXP_NAME=sinpend_kernel4_stride2
#EXP_NAME=sinpend_kernel8_stride4

#EXP_NAME=doupend_kernel2_stride1
#EXP_NAME=doupend_kernel4_stride2
#EXP_NAME=doupend_kernel8_stride4


echo "Training started at: $(date)"

RESULT_DIR=results/${EXP_NAME}-2

rm -rf ${RESULT_DIR}

python main.py \
--config=configs/${EXP_NAME}.py \
--mode=train \
--config.workdir=${RESULT_DIR}

echo "Training finished at: $(date)"
