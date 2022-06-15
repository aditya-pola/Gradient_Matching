All the tests are contained within "all_the_tests.ipynb"

`naive_testing.py` is now complete.
When run, `naive_testing` finds sunset of size 1000 after every epoch of training the
model on full dataset and reports test accuracy metrics for the top_1000 (cosine based),
random subset and full dataset models.


Location of saving data needs to be changed in the variables "data" and "test_data" for local usage.

ToDo:
[] Store the indices of top 1000 subset and see which samples are recurring
