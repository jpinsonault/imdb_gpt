FROM tf_base

RUN pip3 install --ignore-installed keras requests tensorflow numpy tqdm openai datasets prettytable flask flask-cors simplejson pytz
