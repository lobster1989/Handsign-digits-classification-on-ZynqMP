ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json
OUTDIR=../output
NET_NAME=customcnn
MODEL=../output/quantized_model.h5

echo "-----------------------------------------"
echo "COMPILING MODEL FOR ZCU102.."
echo "-----------------------------------------"

compile() {
      vai_c_tensorflow2 \
            --model           $MODEL \
            --arch            $ARCH \
            --output_dir      $OUTDIR \
            --net_name        $NET_NAME
	}


compile 2>&1 | tee compile.log


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"
