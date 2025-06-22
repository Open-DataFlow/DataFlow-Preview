export HF_ENDPOINT=https://hf-mirror.com
export API_KEY="sk-bpQKw8OdWE1o8U2aDDw705dVa12iiMZv3wSzU9yXMUM7bNFd"
# # ------------------------------ Question ------------------------------#
# # Step 0, Initial Filter
echo -e "\033[32m===== [Step 0] Initial Filter =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/ReasoningPipeline/yaml/SFT/process/MathProblemFilter.yaml --step_name QuestionFilter

# Step 1, Question Synthesis
echo  -e "\033[32m===== [Step 1] Question Synthesis =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/ReasoningPipeline/yaml/SFT/generate/QuestionGenerator.yaml --step_name QuestionGenerator 

# Step 2, Question Correctness Filter
echo -e "\033[32m===== [Step 2] Question Correctness Filter =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/ReasoningPipeline/yaml/SFT/process/MathProblemFilter_step2.yaml --step_name QuestionFilter

# Step 3, Difficulty classification
echo -e "\033[32m===== [Step 3] Difficulty Classification =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/ReasoningPipeline/yaml/SFT/generate/QuestionDifficultyClassifier.yaml --step_name QuestionDifficultyClassifier

# Step 4, Category classification
echo -e "\033[32m===== [Step 4] Category Classification =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/ReasoningPipeline/yaml/SFT/generate/QuestionCategoryClassifier.yaml --step_name QuestionCategoryClassifier

# # ------------------------------ Classifier ------------------------------#
# echo -e "\033[32m===== [Step 4.5] GT classification =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/ReasoningPipeline/yaml/SFT/process/AnswerPipelineRoot.yaml --step_name AnswerPipelineRoot 

# # # ------------------------------ Answer ------------------------------#

# Step 5, Answer generation with GT
echo -e "\033[33m===== [Step 5] Pseudo Answer Generation =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/ReasoningPipeline/yaml/SFT/generate/AnswerGenerator.yaml --step_name AnswerGenerator 

# Step 6, Answer format filter
echo -e "\033[33m===== [Step 6] Answer Format Filter =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/ReasoningPipeline/yaml/SFT/process/ReasonerFormatFilter.yaml --step_name AnswerFormatterFilter 

# Step 7, Answer length filter
echo -e "\033[33m===== [Step 7] Answer Length Filter =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/ReasoningPipeline/yaml/SFT/process/ReasonerLengthFilter.yaml --step_name AnswerTokenLengthFilter 

# Step 9, Answer verification with groundtruth
echo -e "\033[33m===== [Step 9] Answer Verification =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/ReasoningPipeline/yaml/SFT/process/ReasonerAnsSelection.yaml  --step_name AnswerGroundTruthFilter 

# Step 10, deduplication according to ngram
echo -e "\033[33m===== [Step 10] Deduplication (N-gram) =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/ReasoningPipeline/yaml/SFT/process/ReasonerNgramFilter.yaml --step_name AnswerNgramFilter 


# ------------------------------ Answer without GT ------------------------------#

# Step 5, Pseudo answer generation
echo  "\033[33m===== [Step 5] Pseudo Answer Generation =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/ReasoningPipeline/yaml/SFT/generate/PseudoAnswerGenerator.yaml --step_name PseudoAnswerGenerator

# Step 6, Answer format filter
echo  "\033[33m===== [Step 6] Answer Format Filter =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/ReasoningPipeline/yaml/SFT/process/ReasonerFormatFilter_withoutGT.yaml --step_name AnswerFormatterFilter

# Step 7, Answer length filter
echo  "\033[33m===== [Step 7] Answer Length Filter =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/ReasoningPipeline/yaml/SFT/process/ReasonerLengthFilter_withoutGT.yaml --step_name AnswerTokenLengthFilter

# Step 10, deduplication according to ngram
echo  "\033[33m===== [Step 10] Deduplication (N-gram) =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/ReasoningPipeline/yaml/SFT/process/ReasonerNgramFilter_withoutGT.yaml --step_name AnswerNgramFilter