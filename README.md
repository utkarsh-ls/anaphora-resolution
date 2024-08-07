# Anaphora Resolution from Social Media Text in Indian Languages

## Setup correct config
- The project works on various configurations where we can choose between transformer backbone (mBERT or MURIL), and whether to work on all languages of a specific one.
- To configure, just copy the required config file from `/all_configs` to `/src/configs.py`.

## Installation
- `pip install -r requirements.txt`
- Download weights folder from the link provided and save it in root directory of project:  
https://iiitaphyd-my.sharepoint.com/:f:/g/personal/vedansh_mittal_students_iiit_ac_in/EtMOSJUej1ZAlXmpZsxMt4kBUf1fLItRAP8U16agboNC6Q?e=D7aSED
> Since, the transformer weights are large, you may want to selectively download checkpoints of a config and save them into `/saved_ckpts` folder manually. The correct checkpoint names for a particular config is present its respective configs file.

## Running code
> Note: all following commands to be run from `/src` folder.
### Data Cleaning
- To clean data: `python data_cleaner.py`.  
This would save the cleaned data file in `/data/clean`.

### Training
- To train mention_model: `python pl_trainer.py --model mention`.  
The logs would be generated in `/logs` folder.
- To train pair_score model: `python pl_trainer.py --model ps`.  
Make sure to copy the old logs folder of mention model somewhere else to prevent overwriting and copy the suitable checkpoint of mention_model in the path mentioned in `configs.py`. (pair_score model needs trained_mention model to work).

### Evaluation

> Note: Two separate tweets need to be separate by a empty line in the file.

- To model on a single data point from prediction, just write the tweets in the file `inp.txt` and the run  
    ```python eval.py --predict_from_file```
- If the input tweets is a single line then we can also run, like:  
    ```python eval.py --predict "John thinks he will win"```.

- To generate evaluation metric over the test dataset:  
    ```python eval.py --eval_all_file```

## Model training and evaluation details

[Presnetation](https://iiitaphyd-my.sharepoint.com/:p:/r/personal/utkarsh_upadhyay_alumni_iiit_ac_in/_layouts/15/Doc.aspx?sourcedoc=%7B37AF730A-F066-41A1-93F5-FE540E406790%7D&file=IRE%20final%20presentation.pptx&action=edit&mobileredirect=true&DefaultItemOpen=1&login_hint=utkarsh.upadhyay%40alumni.iiit.ac.in&ct=1719846805383&wdOrigin=OFFICECOM-WEB.START.EDGEWORTH&cid=bceada01-efc5-4d5e-92eb-7a84556cf813&wdPreviousSessionSrc=HarmonyWeb&wdPreviousSession=9d64d47d-a627-48b8-837c-c0bef0291cdb)
