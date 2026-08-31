import joblib
import numpy as np
AcademicCompiler = joblib.load("/Users/arahan/Desktop/Synopsys2025/CompilerModels/AcademicCompiler.pkl")
MediaCompiler = joblib.load("/Users/arahan/Desktop/Synopsys2025/CompilerModels/MediaCompiler.pkl")
LiteratureCompiler = joblib.load("/Users/arahan/Desktop/Synopsys2025/CompilerModels/LiteratureCompiler.pkl")
sample = np.array([[0.0, 0.0, 1.0]])

print(max(AcademicCompiler.predict(sample)))
print(AcademicCompiler.predict_proba(sample))
