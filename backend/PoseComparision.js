const { subtract, sqrt, norm, dot, round } = require('mathjs');

const THRESHOLD_PERFECT = 0.8
const THRESHOLD_GOOD = 0.7
const THRESHOLD_BAD = 0.6

const SCORE_PERFECT = 500
const SCORE_GOOD = 300
const SCORE_BAD = 100
const SCORE_X = 0

const KEYPOINT_NUM = 14

// DELETE ME
const COMPARISION_DICT = [0, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13]


exports.compare_poses = (user_pose, actual_pose) => {
    cos = cosine_dist_test(user_pose, actual_pose)
    let results = []

    for(let i = 0; i < KEYPOINT_NUM; i++) {
        if(cos[i] == -1) {
            results.push(-1);
            continue;
        }
        results.push(scoreVal(cos[i]))
    }

    console.log("results", results)
    return results
}

function cosine_dist_test(user_pose , actual_pose) {
    let results = [500]
    console.log("user_pose", user_pose);
    console.log("actual_pose", actual_pose);
    // console.log("b", b)

    for(let i = 1; i < KEYPOINT_NUM; i++) {
        if((user_pose[i][0] == 0 && user_pose[i][1] == 0) || (actual_pose[i][0] == 0 && actual_pose[i][1] == 0)) {
            results.push(-1);
            continue;
        }
        a1 = subtract(user_pose[i], user_pose[COMPARISION_DICT[i]])
        b1 = subtract(actual_pose[i], actual_pose[COMPARISION_DICT[i]])

        res = 1.0 - sqrt(1.0 - cosine_sim(a1, b1))
        results.push(res)
    }

    return round(results, 2)
}

function cosine_sim(a, b) {
    return dot(a, b) / (norm(a) * norm(b))
}

function scoreVal(score) {
    if(score >= THRESHOLD_PERFECT) {
        return SCORE_PERFECT
    }else if(score >= THRESHOLD_GOOD) {
        return SCORE_GOOD
    }else if(score >= THRESHOLD_BAD) {
        return SCORE_BAD
    }else{
        return SCORE_X
    }
        
}
