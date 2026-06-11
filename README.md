# Salary Prediction with Linear Regression

## 📋 Project Description

This project builds a machine learning model that predicts employee salaries based on years of experience using **Linear Regression**. The project includes both a scikit-learn implementation and a **custom linear regression implementation** to understand how the algorithm works internally.

## 🎯 Objectives

- ✅ Develop a predictive model for employee salary estimation
- ✅ Implement linear regression from scratch to understand the mathematics
- ✅ Compare custom implementation with scikit-learn
- ✅ Demonstrate data analysis and machine learning fundamentals
- ✅ Achieve high model accuracy on test data

## 📊 Dataset

The project uses the **Salary_dataset.csv** containing:
- **30 samples** of employee records
- **Years of Experience**: ranging from 1.2 to 10.6 years
- **Salary**: ranging from $37,732 to $122,392
- **Data split**: 67% training, 33% testing

## 🛠️ Technologies & Libraries

- **Python 3**
- **Jupyter Notebook** (Google Colab compatible)
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning tools
- **Matplotlib**: Data visualization

## 📁 Project Structure

```
Salary_Prediction/
├── M2.ipynb                 # Main Jupyter notebook with implementation
├── Salary_dataset.csv       # Employee salary dataset
└── README.md               # This file
```

## 🔧 Custom Linear Regression Implementation

The project includes a custom `M2` class that implements linear regression from scratch:

```python
class M2:
    def __init__(self):
        self.m = 0      # Slope
        self.b = 0      # Intercept
    
    def train(self, X_train, Y_train):
        # Calculates slope (m) and intercept (b) using the least squares method
        num = 0
        den = 0
        x_bar = X_train.mean()
        y_bar = Y_train.mean()
        
        for i in range(len(X_train)):
            num += (X_train[i] - x_bar) * (Y_train[i] - y_bar)
            den += (X_train[i] - x_bar) ** 2
        
        self.m = num / den
        self.b = y_bar - self.m * x_bar
    
    def predict(self, X):
        return self.m * X + self.b
```

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **R² Score (Test Data)** | **0.9553** |
| **Slope (m)** | ~9,426 |
| **Intercept (b)** | ~24,382 |
| **Prediction Formula** | `Salary = 9426 × YearsExperience + 24382` |

### Interpretation
- For each additional year of experience, the salary increases by approximately **$9,426**
- Base salary (intercept) starts at approximately **$24,382**
- The model explains **95.53%** of the variance in salary data

## 💻 Usage

### 1. Clone the Repository
```bash
git clone https://github.com/prameela-rani-patnaik/Salary_Prediction.git
cd Salary_Prediction
```

### 2. Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib jupyter
```

### 3. Run the Notebook
```bash
jupyter notebook M2.ipynb
```

Or open directly in Google Colab:
- Click the "Open in Colab" link in the notebook

### 4. Execute All Cells
Run through all cells sequentially to:
- Load and explore the dataset
- Train the custom linear regression model
- Make predictions
- Visualize results
- Calculate R² score

## 📚 Workflow

1. **Data Loading**: Import dataset using Pandas
2. **Data Exploration**: Check dataset shape and statistics
3. **Data Preparation**: Split into features (X) and target (Y)
4. **Train-Test Split**: 67% training, 33% testing
5. **Model Training**: Train custom M2 model on training data
6. **Predictions**: Generate predictions on test data
7. **Evaluation**: Calculate R² score to assess model performance
8. **Visualization**: Plot training data with regression line

## 📊 Visualization

The project generates a scatter plot showing:
- Training data points
- Linear regression best-fit line
- Clear positive correlation between experience and salary

## 🔍 Key Learnings

- **Linear Regression Formula**: `y = mx + b`
- **Least Squares Method**: Minimizes sum of squared errors
- **Model Evaluation**: R² score measures goodness of fit
- **Data Preprocessing**: Essential for model accuracy
- **Train-Test Split**: Crucial for unbiased performance evaluation

## 🚀 Future Enhancements

- [ ] Add multiple features (education, position, department)
- [ ] Implement polynomial regression
- [ ] Apply cross-validation for robust evaluation
- [ ] Compare with other regression algorithms (Ridge, Lasso)
- [ ] Deploy as a web application (Flask/Django)
- [ ] Add residual analysis and diagnostics
- [ ] Implement feature scaling and normalization

## 📝 Notes

- The dataset shows a strong linear relationship between years of experience and salary
- The high R² score (0.9553) indicates an excellent fit
- This implementation is educational and demonstrates ML fundamentals
- The custom implementation helps understand the mathematical foundations of linear regression

## 👤 Author

[prameela-rani-patnaik](https://github.com/prameela-rani-patnaik)

## 📄 License

This project is open source. Feel free to use, modify, and distribute as needed.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or suggestions
- Submit pull requests with improvements
- Share feedback and ideas

## 📧 Contact

For questions or discussions about this project, feel free to reach out through GitHub.

---

**Last Updated**: June 2026  
**Status**: ✅ Complete and Functional
