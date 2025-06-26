# # ------------------------------ Question ------------------------------#
# # Step 0, Initial Clustering and filter
echo -e "\033[32m===== [Step 0] Filter  =====\033[0m"
python pipeline_step.py --yaml_path dataflow/scripts/pipelines/AgenticRAGPipeline/yaml/process/ContentChooser.yaml --step_name ContentChooser

#Step 1, Prompt Synthesis
#echo  -e "\033[32m===== [Step 1] Prompt Synthesis =====\033[0m"
#python pipeline_step.py --yaml_path dataflow/scripts/pipelines/AgenticRAGPipeline/yaml/generate/AutoPromptGenerator.yaml --step_name AutoPromptGenerator 

# Step 2, QA Synthesis
#echo -e "\033[32m===== [Step 2] QA Synthesis =====\033[0m"
#python pipeline_step.py --yaml_path dataflow/scripts/pipelines/AgenticRAGPipeline/yaml/generate/QAGenerator.yaml --step_name QAGenerator

# Step 3, QA Scorer
#echo -e "\033[32m===== [Step 3] QA Scorer =====\033[0m"
#python pipeline_step.py --yaml_path dataflow/scripts/pipelines/AgenticRAGPipeline/yaml/generate/QAScorer.yaml --step_name QAScorer