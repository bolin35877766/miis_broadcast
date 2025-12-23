def convert_millisecs(millis):
    seconds = millis/1000%60
    minutes = int(millis/(1000*60)%60)
    hours = int(millis/(1000*3600)%24)
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"