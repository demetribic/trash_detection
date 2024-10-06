
from predict import predict_main
from points import User

def run_predict_and_update(img, curr_score):
    user = User()
    final_label = predict_main(img)
    curr_score =  user.detect_trash(final_label, curr_score)
    return final_label, curr_score