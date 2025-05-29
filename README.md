# E-commerce Customer Satisfaction Prediction

*A comprehensive machine learning case study for predicting customer review ratings in e-commerce platforms*

## 📋 Project Overview

In today's digital landscape, e-commerce has become an integral part of our daily lives, serving as the bridge connecting countless sellers with customers worldwide. For e-commerce businesses to thrive, delivering exceptional customer service is paramount. But how do we measure service quality? The answer lies in **customer satisfaction**.

Customer feedback and review ratings serve as crucial indicators of business performance. When customers are dissatisfied, businesses must focus on improving service quality to retain customers and drive growth. This project addresses a critical challenge in e-commerce: **predicting customer review ratings before customers actually provide them**.

## 🎯 Business Problem

Traditional e-commerce platforms send feedback forms to customers after product delivery, requesting ratings (1-5 stars) and written reviews. However, many customers don't provide feedback, leaving businesses without crucial satisfaction insights. This project aims to solve:

- **Primary Question**: Can we predict the review rating a customer would give before they actually provide it?
- **Business Impact**: Enable proactive customer service improvements and predict satisfaction for non-responsive customers

## 📊 Dataset

**Source**: [Olist Brazilian E-commerce Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce) from Kaggle

**About Olist**: A Brazilian e-commerce platform connecting small businesses across Brazil with customers through a single contract.

**Dataset Details**:
- **Time Period**: 2016-2018
- **Records**: 100k+ order information
- **Files**: 9 CSV files containing customer, seller, product, payment, and review data
- **Final Dataset**: 113,105 instances after preprocessing

## 🔬 Machine Learning Approach

### Problem Formulation
- **Type**: Supervised Multi-class Classification
- **Target Variable**: Review scores (1-5 stars)
- **Classes**: 5 discrete rating levels
- **Challenge**: Highly imbalanced dataset with J-shaped distribution

### Key Constraints
- **Latency**: No strict real-time requirements, but predictions needed before customer feedback
- **Business Priority**: Minimizing misclassification of low ratings (1,2,3) to prevent customer loss

### Evaluation Metrics
- **Primary**: Macro F1 Score
- **Secondary**: Multi-class Confusion Matrix
- **Additional**: Balanced Accuracy, Multi-class Log Loss

## 🛠️ Technical Implementation

### Tools & Technologies
- **Programming Language**: Python
- **Environment**: Jupyter Notebook
- **Core Libraries**:
  - **Data Processing**: pandas, numpy
  - **Machine Learning**: scikit-learn, imbalanced-learn
  - **Visualization**: matplotlib, seaborn
  - **Advanced ML**: LightGBM, XGBoost

### Data Processing Pipeline
1. **Data Integration**: Merged 9 CSV files using provided schema
2. **Data Cleaning**: Handled null values, removed duplicates, filtered delivered orders
3. **Feature Selection**: Excluded post-delivery information (reviews, comments)
4. **Data Quality**: Retained 96% of original data after cleaning

## 🔍 Exploratory Data Analysis

### Key Findings
- **Class Imbalance**: Severe J-shaped distribution (Rating 5: 57.15%, Rating 2: 3.47%)
- **Geographic Patterns**: São Paulo (SP) dominates both customers (42%) and sellers (58.5%)
- **Product Categories**: bed_bath_table most frequent across all ratings
- **Payment Methods**: Credit card preferred (73.8%), followed by boleto (19%)

### Statistical Insights
- No single feature strongly correlates with review scores
- Payment type and review score are statistically dependent (χ² test, p < 0.001)
- Geographic proximity between sellers and customers affects satisfaction

## ⚙️ Feature Engineering

### Time-Based Features
- **Delivery Performance**: `actual_time`, `estimated_time`, `diff_actual_estimated`
- **Processing Times**: `diff_purchased_approved`, `diff_purchased_courier`
- **Temporal Patterns**: delivery day/hour, purchase day/hour

### Distance-Based Features
- **Geographic Distance**: Haversine distance between seller and customer locations
- **Delivery Speed**: `speed = distance/actual_time`
- **Product Dimensions**: `size = length × breadth × height`

### Advanced Features
- **Similarity Metrics**: Seller-customer compatibility scores
- **Market Share**: Seller and customer activity ratios
- **Binary Indicators**: late_shipping, same_state, same_city

## 🤖 Model Development

### Approach Evolution

#### 1. Direct Multi-class Classification
- **Models Tested**: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, LightGBM, XGBoost
- **Challenge**: Poor performance due to severe class imbalance
- **Best Result**: LightGBM with Macro F1 = 0.50

#### 2. Imbalance Handling Techniques
- **Random Oversampling**: No significant improvement
- **RUSBoost**: Underperformed on minority classes
- **SMOTE**: Limited effectiveness due to feature characteristics

#### 3. Hierarchical Classification (Final Approach)
**Model 1**: Binary Classification (Rating 5 vs Others)
- **Best Model**: Logistic Regression
- **Performance**: F1 Score = 0.61 (Test)
- **Advantage**: Handles major class effectively

**Model 2**: Multi-class Classification (Ratings 1,2,3,4)
- **Approach**: Custom Ensemble with Meta-learning
- **Architecture**: Multiple base learners + meta-classifier
- **Performance**: F1 Score = 0.23 (Test)

### Final Model Architecture
```
Input Features → Model 1 (Binary) → Rating 5 OR → Model 2 (Multi-class) → Ratings 1,2,3,4
```

## 📈 Results & Performance

### Model Performance
- **Binary Classification**: 61% F1 Score (Rating 5 vs Others)
- **Multi-class Classification**: 23% F1 Score (Ratings 1-4)
- **Overall Challenge**: Severe class imbalance limits performance on minority classes

### Business Impact
- Successfully identifies highly satisfied customers (Rating 5)
- Provides framework for proactive customer service intervention
- Enables targeted improvements for low-satisfaction predictions

## 🚀 Deployment

The model has been deployed using **Flask** web framework, providing a user-friendly interface for real-time predictions.

**Features**:
- Real-time rating prediction
- Input validation and preprocessing
- Confidence scores for predictions
- Business-friendly output format

## 🔮 Future Improvements

### Technical Enhancements
1. **Advanced Feature Engineering**: Incorporate customer demographics (age, gender, financial status)
2. **Deep Learning**: Explore neural networks for complex pattern recognition
3. **Ensemble Methods**: Develop more sophisticated voting mechanisms
4. **Time Series Analysis**: Leverage temporal patterns in customer behavior

### Data Augmentation
- **External Data**: Weather, economic indicators, seasonal trends
- **Social Media**: Sentiment analysis from customer social profiles
- **Behavioral Data**: Click-through rates, browsing patterns, cart abandonment

### Model Optimization
- **Cost-Sensitive Learning**: Assign higher costs to minority class misclassifications
- **Active Learning**: Iteratively improve model with strategic data collection
- **Multi-objective Optimization**: Balance accuracy across all rating levels

## 📚 References & Resources

- [Original Kaggle Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce)
- [Customer Satisfaction Research](https://www.researchgate.net/publication/323111412_The_effects_of_customer_satisfaction_with_e-commerce_system)
- [Imbalanced Learning Techniques](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/)
- [E-commerce Analytics](https://towardsdatascience.com/using-data-science-to-predict-negative-customer-reviews-2abbdfbf3d82)


