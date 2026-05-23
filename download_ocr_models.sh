#!/bin/bash
# Script to download PaddleOCR inference models for fully offline execution

mkdir -p models
cd models

echo "Downloading PaddleOCR Detection Model..."
curl -L -o en_PP-OCRv3_det_infer.tar https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar
tar -xf en_PP-OCRv3_det_infer.tar
rm en_PP-OCRv3_det_infer.tar

echo "Downloading PaddleOCR Recognition Model..."
curl -L -o en_PP-OCRv4_rec_infer.tar https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar
tar -xf en_PP-OCRv4_rec_infer.tar
rm en_PP-OCRv4_rec_infer.tar

echo "Downloading PaddleOCR Angle Classification Model..."
curl -L -o ch_ppocr_mobile_v2.0_cls_infer.tar https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar -xf ch_ppocr_mobile_v2.0_cls_infer.tar
rm ch_ppocr_mobile_v2.0_cls_infer.tar

echo "✅ All OCR models downloaded to ./models successfully!"
