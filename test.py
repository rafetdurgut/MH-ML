from Model  import *

A=[5.3,6.57,6.76, 11.01, 7.93,7.48,7.71,7.42, 7.09, 12.46, 15.25, 14.98, 14.65, 15.11, 15.11, 17.98, 13.96,14.96]
B=[6.6288,7.6108,7.2148,7.6576,7.3836,7.1356,9.7152, 6.9292,6.9772,13.3752, 14.8012, 14.8432, 14.6104, 14.7804, 14.9704,17.0808, 14.7388, 15.2248]
print(mean_squared_error(A,B, squared=True))