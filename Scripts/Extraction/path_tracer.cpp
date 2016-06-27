// Dependencies: Install CMake and OpenCV (for C++)

#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dirent.h>

//#include <cv.h>
//#include <highgui.h>

using namespace cv;
using namespace std;

/** Example usage: ./img_trace squiggle.png **/

/* Input: filename, Output: bitmap matrix of intensities + image size */
static void fill_bitmap(const char *filename, vector<vector<int> > &bitmap, int &width, int &height) {

    Mat image;
    image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    width = image.size().width; //number of cols
    height = image.size().height;//number of rows
    bitmap = vector< vector<int> > (height, vector<int>(width));

    stringstream ss;
    //cout << image;
    ss << image;
    char trash[3];
    ss.get(trash, 2);

    for(int i = 0; i < height; i++) {
       for(int j = 0; j < width; j++) {
          ss >> bitmap[i][j];
          ss.get(trash, 2);
       }
    }
}


/* Input: bitmap matrix, Output: time series */
static void get_ts(const vector<vector<int> > &bitmap, vector<int> &ts, double &ts_loss, int width, int height) {

    /* initialize constants */
    double alpha = 0.5;               // intensity of the point
    double gamma = 1 - alpha;  // deviation

    /* initialize vectors */
    vector<int> path_row (width, 0);
    vector<double> loss_row (width, 0);
    vector<vector<int> > path (height, path_row);
    vector<vector<double> > loss (height, loss_row);

    for (int r = 0; r < height; r++) {

        /* intensity */
        for (int c = 1; c < width - 1; c++) {
            loss[r][c] = -alpha * bitmap[r][c];
        }

        /* if we are on the first row, don't look for a previous row */
        if (r == 0) {
            continue;
        }

        /* distance */
        for (int c = 1; c < width - 1; c++) {
            int best_cc;
            double best_score = DBL_MAX;
            for (int cc = 1; cc < width - 1; cc++) {
                double score = loss[r - 1][cc] + gamma * (cc - c) * (cc - c);
                if (score < best_score) {
                    best_cc = cc;
                    best_score = score;
                }
            }
            loss[r][c] += best_score;
            path[r][c] = best_cc;
        }
    }

    /* initialize ts and ts_loss */
    ts.resize(height);
    ts_loss = DBL_MAX;

    /* find best ending point */
    for (int c = 1; c < width - 1; c++) {
        if (loss[height - 1][c] < ts_loss) {
            ts_loss = loss[height - 1][c];
            ts[height - 1] = c;
        }
    }

    /* iterate back through path */
    for (int i = height - 1; i > 0; i--) {
        ts[i - 1] = path[i][ts[i]];
    }
}

/* Overlays the resulting time series on the raw image, for convenient viewing */
static void overlay(const char *filename, const vector<vector<int> > &bitmap, const vector<int>& ts, int width, int height) {
    Mat image;
    image = imread(filename, 1); //open with RGB

    for(int ii = 0; ii < ts.size(); ii++){
        Vec3b color = image.at<Vec3b>(Point(ts[ii],ii));
        color.val[0] = 0;
        color.val[1] = 0;
        color.val[2] = 255;
        image.at<Vec3b>(Point(ts[ii], ii)) = color;
    }

    //imwrite( "/" + filename + "_ts.png", image);
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    waitKey(20);
}

// Note: Modify the directory path to fit your needs
int main(int argc, char* argv[]) {

    DIR *dir;
    struct dirent *ent;
    char* filename;

    if ((dir = opendir ("/Users/Jason/Desktop/WaterfallPlots/")) != NULL) {
         /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            printf("%s\n", ent->d_name);
            string file = ent->d_name;
            if (file[0] == '.') { // || file.substr(file.length() - 5, 3) != ".png") {
                continue;
            }

            string prefix = "/Users/Jason/Desktop/WaterfallPlots/";
            string temp = prefix + file;
            filename = (char *)malloc(temp.size() + 1);
            memcpy(filename, temp.c_str(), temp.size() + 1);

            int width, height;
            double ts_loss;
            vector<int> ts;
            vector<vector<int> > bitmap;

            fill_bitmap(filename, bitmap, width, height);
            get_ts(bitmap, ts, ts_loss, width, height);
            overlay(filename, bitmap, ts, width, height);

            for (int i = 0; i < height; i++) {
                cout << " " << ts[i];
            }
            cout << ": " << ts_loss << endl;
        }
        closedir (dir);
    }

    return 0;
}

