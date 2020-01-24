# Pump it Up: Data Mining the Water Table

## Team Members

Sam Videlock, Edna Fernandes and Anup Sebastian


## Project Goals/Overview

The purpose of this project was to use data from Taarifa and the Tanzanian Ministry of Water and predict which pumps are functional, which need repair and which are non functional. This helps optimize Tanzanian maintenance operations and ensure that clean, potable water is available to communities across Tanzania.

In our analysis we were able to identify a smaller subset of non functioning wells that could be prioritized for restoration based on the insights. The analysis can be found on the Power BI file "Data_Analysis.pbix".

The final presentation can be found in the file 'Final_Presentation.pdf' or from this link: https://www.canva.com/design/DADxqPE0Yxg/Hurc4uIQFubGl3QY-9lOyA/view?utm_content=DADxqPE0Yxg&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton



## README Summary

In this README we will discuss our process for data cleaning, our insights in the dataset and our modeling process.



### Data Cleaning

The original training dataset is found on 'train_set_values.csv', 'train_set_labels.csv'. The test set can be found on 'test_set_values.csv'. The data comes from the drivendata.org for the competition Pump it Up: Data Mining the Water Table.

For our cleanup, we created a preprocessor file (preprocessor.py). In the preprocessor we have the following steps:
    - Merged the 'train_set_values.csv', 'train_set_labels.csv'.
    - Replaced all the null values with the mode.
    - Dropped some columns that seemed repretitive.
    - Target encoded our features
    

### Models

    Binary Model
      - Can be found in the file "Binary_Simplified_Model.ipynb".
      - Combined the targets "functional" and "functional needs repair" into one.
      - Due to the small amount of "functional needs repair" target, the simplification was still able to get relatively good       results (79% accuracy).


    Final Model...
      - Can be found in the file 
      - Had the three different targets ("functional", "functional needs repair" and "non functional").
      - We got an accuracy score of
