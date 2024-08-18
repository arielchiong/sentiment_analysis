### Python3

review files are stored in /data directory.

make sure to create directory /models this is where the train models will be stored.

### Script
Install dependencies
```
pip install tensorflow pandas scikit-learn numpy jinja2
```

Train the Ai to understand the positive and negative feedback and comments. 
This will take some time depending on your PC/Laptop processors. I suggest to run this and just leave it.
```
python3 model.py 
```

Once model.py is done we can now run the web application.
Web framework to input some feedbacks.

```
python3 app.py
```
