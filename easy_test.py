import numpy as np
import pandas as pd

def easy_test():

    data = {
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35],
        "Score": [88.5, 92.3, 85.0]
    }
    df = pd.DataFrame(data)
    latex_code = df.to_latex(index=False, caption="Example Table", label="table:example")
    print(latex_code)
if __name__ == "__main__":
    easy_test()