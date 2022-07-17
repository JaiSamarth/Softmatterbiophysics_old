/*
This code 
takes in .bmp image data
from phase contrast microscopy

identifies contours of elliptical guvs

fits an ellipse over contour

saves ellipse data for each frame in a 
csv file

-jsama
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

using namespace cv;
using namespace std;

// constant pi
const double pi = M_PI;
// gui input
double gui_in[6] = {1.0}; // x0, y0, rin, rou, ecc, phi

void detect_click(int event, int x, int y, int flags, void *data)
{
    // copy image
    string *image_path = (string *)data;
    Mat imc = imread(*image_path);
    // left click
    if (event == EVENT_LBUTTONDOWN)
    {
        // get click coords
        gui_in[0] = x;
        gui_in[1] = y;
        // draw ellipse
        ellipse(imc,
         Point(gui_in[0], gui_in[1]),
          Size(gui_in[2], gui_in[2] * sqrt(1.0 - pow(gui_in[4],2))),
          gui_in[5],
          0,
          360,
          CV_RGB(0,255,0),
          1,
          LINE_AA);
        ellipse(imc,
         Point(gui_in[0], gui_in[1]),
          Size(gui_in[3], gui_in[3] * sqrt(1.0 - pow(gui_in[4],2))),
          gui_in[5],
          0,
          360,
          CV_RGB(255,0,0),
          1,
          LINE_AA);
        // update window
        imshow("select interest area", imc);
    }  
}

void slider(int val, void *data)
{
    // copy image
    string *image_path = (string *)data;
    Mat imc = imread(*image_path);
    // get trackbar data
    gui_in[2] = getTrackbarPos("rin", "select interest area");
    gui_in[3] = getTrackbarPos("rou", "select interest area");
    gui_in[4] = getTrackbarPos("ecc*100", "select interest area") * 0.01;
    gui_in[5] = getTrackbarPos("phi", "select interest area");
    // draw ellipse
    ellipse(imc,
        Point(gui_in[0], gui_in[1]),
        Size(gui_in[2], gui_in[2] * sqrt(1.0 - pow(gui_in[4],2))),
        gui_in[5],
        0,
        360,
        CV_RGB(0,255,0),
        1,
        LINE_AA);
    ellipse(imc,
        Point(gui_in[0], gui_in[1]),
        Size(gui_in[3], gui_in[3] * sqrt(1.0 - pow(gui_in[4],2))),
        gui_in[5],
        0,
        360,
        CV_RGB(255,0,0),
        1,
        LINE_AA);
    // update window
    imshow("select interest area", imc);
}

void interest_area(Mat img, string path)
{
    // named window
    namedWindow("select interest area");
    // set callback function
    setMouseCallback("select interest area", detect_click, &path);
    // view image
    imshow("select interest area", img);
    // create trackbar
    createTrackbar("phi", "select interest area", 0, 360, slider, &path);
    createTrackbar("ecc*100", "select interest area", 0, 100, slider, &path);
    createTrackbar("rin", "select interest area", 0, min(img.rows, img.cols), slider, &path);
    createTrackbar("rou", "select interest area", 0, min(img.rows, img.cols), slider, &path);
    // close window
    waitKey(0);
    destroyWindow("select interest area");
}

void apply_filter(Mat *img, int l)
{
    // copy image
    Mat imf = *img;
    // apply clahe
    equalizeHist(imf, imf);
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(10);
    clahe->apply(imf, imf);
    // apply blur
    blur(imf, imf, Size(5,5), Point(-1, -1));
    GaussianBlur(imf, imf, Size(5,5), 0, 0);
    if (l == 1)
    {
        // laplace filter
        Mat lap;
        Laplacian(imf, lap, CV_16S, 3, 2, 0, BORDER_DEFAULT);
        convertScaleAbs(lap, imf);
    }
    else
    {
        // sobel filter
        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;
        Sobel(imf, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
        Sobel(imf, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
        // converting back to CV_8U
        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);
        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, imf);
    }
}

vector<double> linspacef(double a, double b, size_t n)
{
    double spacing = (b-a)/ (double)(n-1);
    vector<double> linspace(n);
    vector<double>::iterator itr;
    double val;
    for (itr = linspace.begin(), val = a; itr != linspace.end(); itr++, val += spacing)
    {
        *itr = val;
    }
    return linspace;
}

void detect_contour(Mat *img, vector<int> &contour_x, vector<int> &contour_y)
{
    int n = 1.5 * (gui_in[2] + gui_in[3]); // set angular resolution
    int m = 2 * (gui_in[3] - gui_in[2]); // set radial resolution
    Mat img_array = *img;
    vector<double> theta = linspacef(0, 2*pi, n);
    vector<double> radius = linspacef(gui_in[2], gui_in[3], m);
    int polar_img[n][m]; // polar image
    // scan interest area
    for (int i = 0; i < n; i++)
    {
        uchar max = 0;
        contour_x.push_back(0);
        contour_y.push_back(0);
        for (int j = 0; j < m; j++)
        {
            // ellipse points
            double ex = radius[j] * cos(theta[i]);
            double ey = ((radius[j] * sqrt(1.0 - pow(gui_in[4],2))) * sin(theta[i]));
            // rotate coordinates
            int x = (int)round(
                ex * cos(gui_in[5] * pi/180.0) - ey * sin(gui_in[5] * pi/180.0) + gui_in[0]
                );
            int y = (int)round(
                ex * sin(gui_in[5] * pi/180.0) + ey * cos(gui_in[5] * pi/180.0) + gui_in[1]
                );
            // max peak
            if (img_array.at<uchar>(y,x) >= max)
            {
                max = img_array.at<uchar>(y,x);
                contour_x.back() = x;
                contour_y.back() = y;
            }  
        }
    }
}

template <typename t>
t vmean(vector<t> data)
{
    t sum = 0;
    for (size_t i = 0; i < data.size(); i++)
    {
        sum += data[i];
    }
    return (t)(sum/data.size());
}

void direct_ellipse(double ellipse_in[5], vector<int> &contour_x, vector<int> &contour_y)
{   
    // centroid of data points
    double xm, ym;
    xm = vmean(contour_x);
    ym = vmean(contour_y);
    // reduced coordinates
    Eigen::VectorXd x_cord(contour_x.size());
    Eigen::VectorXd y_cord(contour_x.size());
    for (size_t i = 0; i < contour_x.size(); i++)
    {
        x_cord(i) = contour_x[i] - xm;
        y_cord(i) = contour_y[i] - ym;
    }
    // design matrix
    Eigen::MatrixXd D1(contour_x.size(), 3);
    Eigen::MatrixXd D2(contour_x.size(), 3);
    D1.col(0) = x_cord.array().pow(2);
    D1.col(1) = x_cord.array() * y_cord.array();
    D1.col(2) = y_cord.array().pow(2);
    D2.col(0) = x_cord;
    D2.col(1) = y_cord;
    D2.col(2) = Eigen::VectorXd::Ones(contour_x.size());
    // scatter matrix
    Eigen::MatrixXd S1(3,3);
    Eigen::MatrixXd S2(3,3);
    Eigen::MatrixXd S3(3,3);
    S1 = D1.transpose() * D1;
    S2 = D1.transpose() * D2;
    S3 = D2.transpose() * D2;
    // constraint
    Eigen::MatrixXd C1(3,3);
    C1 << 0.0,0.0,2.0,0.0,-1.0,0.0,2.0,0.0,0.0;
    // reduced scatter matrix
    Eigen::MatrixXd T = - S3.inverse()*S2.transpose();
    Eigen::MatrixXd M(3,3);
    M = C1.inverse() * (S1 + S2*T);
    // solve eigen values/vectors
    Eigen::EigenSolver<Eigen::MatrixXd> es(M);
    Eigen::MatrixXd eigvec = es.eigenvectors().real();
    // condition
    Eigen::VectorXd cond = 4.0*(eigvec.row(0).array()*eigvec.row(2).array()) - eigvec.row(1).array().pow(2);
    // min pos eigval
    Eigen::VectorXd a1;
    for (int i = 0; i < 3; i++)
    {
        if (cond(i)>0)
        {
            a1 = eigvec.col(i);
            break;
        }
        
    }
    // a2
    Eigen::VectorXd a2 = T * a1;
    // a
    Eigen::VectorXd A(6);
    A << a1, a2;
    // absolute a
    A(5) = A(5) + A(0) * xm * xm + A(1) * xm * ym + A(2) * ym * ym - A(3) * xm - A(4) * ym;
    A(3) = A(3) - 2.0 * xm * A(0) - ym * A(1);
    A(4) = A(4) - 2.0 * ym * A(2) - xm * A(1);
    A = A/A.norm();
    // algebric to polar
    double a = A(0);
    double b = A(1)/2.0;
    double c = A(2);
    double d = A(3)/2.0;
    double f = A(4)/2.0;
    double g = A(5);
    double det = b*b - a*c;
    ellipse_in[0] = (c*d - b*f)/det;
    ellipse_in[1] = (a*f - b*d)/det;
    double num = 2.0*(a*f*f+c*d*d+g*b*b-2.0*b*d*f-a*c*g);
    double fac = sqrt(pow((a - c), 2) + 4*b*b);
    double ap = sqrt(num / det / (fac - a - c));
    double bp = sqrt(num / det / (-fac - a - c));
    bool width_gt_height = true;
    if (ap<bp)
    {
        width_gt_height = false;
        swap(ap, bp);
    }
    double phi = 0.5 * atan((2.*b) / (a - c));
    if(a>c)
    {
        phi += pi/2;
    }
    if(width_gt_height == false)
    {
        phi += pi/2;
        phi = fmod(phi,pi);
    }
    ellipse_in[2] = ap;
    ellipse_in[3] = bp;
    ellipse_in[4] = phi * 180.0/pi;
}

string padding(int s, int digits)
{
    string str = to_string(s);
    int pad = digits - (int)str.size();
    string name = string(pad, '0').append(str);
    return name;
}

double deviation(double ellipse_in[5], vector<int> &contour_x, vector<int> &contour_y)
{
    // store deviation
    double dev = 0.0;
    // read in ellipse fit
    double xc = ellipse_in[0];
    double yc = ellipse_in[1];
    double ap = ellipse_in[2];
    double bp = ellipse_in[3];
    double ph = ellipse_in[4];
    // theta resolution
    int n = contour_x.size();
    vector<double> theta = linspacef(0, 2*pi, n);
    for (size_t i = 0; i < n; i++)
    {
        // contour points
        double x = contour_x[i];
        double y = contour_y[i];
        // ellipse points
        double ex = ap * cos(theta[i]);
        double ey = bp * sin(theta[i]);
        // rotate and transform
        double exr = ex*cos(ph*pi/180.0) - ey*sin(ph*pi/180.0) + xc;
        double eyr = ex*sin(ph*pi/180.0) + ey*cos(ph*pi/180.0) + yc;
        // square difference
        dev += pow(x - exr, 2) + pow(y - eyr, 2);
    }
    return sqrt(dev/(double) n);
}

int main(){
    // directory path
    string file_path;
    cout << "enter folder path with / at the end : " << endl;
    cin >> file_path;
    // first frame number
    int first_frame;
    cout << "enter first frame number : " << endl;
    cin >> first_frame;
    // last frame number
    int final_frame;
    cout << "enter final frame number : " << endl;
    cin >> final_frame;
    // number of digits
    int digits;
    cout << "enter number of digits in filename : " << endl;
    cin >> digits;
    // laplace or sobel
    int l;
    cout << "enter 1 if laplace, 0 if sobel" << endl;
    cin >> l;
    // deviation upper limit
    double d;
    cout << "enter deviation upper limit" << endl;
    cin >> d;
    string image_path = file_path + padding(first_frame, digits) + ".bmp";
    // read image
    Mat img = imread(image_path, IMREAD_GRAYSCALE);
    // if fail
    if (img.empty())
    {
        cout << "error in opening image" << endl;
        return 1;
    }
    interest_area(img, image_path);
    bool recenter = false;
    namedWindow("output");
    // store data
    ofstream saved_data;
    saved_data.open("pc_fit.csv");
    saved_data << "t,x0,y0,a,b,phi\n";
    for (size_t i = first_frame; i < final_frame; i++)
    {
        // open image
        image_path = file_path + padding(i, digits) + ".bmp";
        Mat img = imread(image_path, IMREAD_GRAYSCALE);
        if (img.empty())
        {
            cout << "error in opening image : " << i << endl;
        }
        else
        {
            // if previous was rejected
            if (recenter == true)
            {
                interest_area(img, image_path);
                recenter = false;
            }
            // apply filters
            apply_filter(&img, l);

            // detect contour
            vector<int> contour_x;
            vector<int> contour_y;
            detect_contour(&img, contour_x, contour_y);

            // fit ellipse
            double ellipse_in[5]; // x0, y0, a, b, phi
            direct_ellipse(ellipse_in, contour_x, contour_y);

            // view 
            Mat imc = imread(image_path, IMREAD_COLOR);

            // contour
            // view detected contour
            for (size_t i = 0; i < contour_x.size(); i++)
            {
                imc.at<Vec3b>(contour_y[i], contour_x[i])[0] = 0;
                imc.at<Vec3b>(contour_y[i], contour_x[i])[1] = 0;
                imc.at<Vec3b>(contour_y[i], contour_x[i])[2] = 255;
            }

            // ellipse fit in discrete form
            // view fitted ellipse in discrete form
            double xc = ellipse_in[0];
            double yc = ellipse_in[1];
            double ap = ellipse_in[2];
            double bp = ellipse_in[3];
            double ph = ellipse_in[4];
            int n = contour_x.size();
            vector<double> theta = linspacef(0, 2*pi, n);
            for (size_t i = 0; i < n; i++)
            {
                double x = contour_x[i];
                double y = contour_y[i];
                double ex = ap * cos(theta[i]);
                double ey = bp * sin(theta[i]);
                double exr = ex*cos(ph*pi/180.0) - ey*sin(ph*pi/180.0) + xc;
                double eyr = ex*sin(ph*pi/180.0) + ey*cos(ph*pi/180.0) + yc;
                imc.at<Vec3b>(eyr, exr)[0] = 255;
                imc.at<Vec3b>(eyr, exr)[1] = 0;
                imc.at<Vec3b>(eyr, exr)[2] = 0;
            }

            // ellipse fit
            // view fitted ellipse in curve form
            // ellipse(imc,
            // Point(ellipse_in[0],
            // ellipse_in[1]), 
            // Size(ellipse_in[2],
            // ellipse_in[3]),
            // ellipse_in[4],
            // 0,
            // 360, 
            // Scalar(0,255,0),
            // 1,
            // LINE_AA);

            // interest area
            // visualize interest area for debugging
            // ellipse(imc,
            // Point(gui_in[0], gui_in[1]),
            // Size(gui_in[2], gui_in[2] * sqrt(1.0 - pow(gui_in[4],2))),
            // gui_in[5],
            // 0,
            // 360,
            // CV_RGB(0,255,0),
            // 1,
            // LINE_AA);
            // ellipse(imc,
            // Point(gui_in[0], gui_in[1]),
            // Size(gui_in[3], gui_in[3] * sqrt(1.0 - pow(gui_in[4],2))),
            // gui_in[5],
            // 0,
            // 360,
            // CV_RGB(255,0,0),
            // 1,
            // LINE_AA);

            // deviation
            // output contour deviation as 
            //text on image for debugging
            putText(imc,
            to_string(deviation(ellipse_in, contour_x, contour_y)),
            Point(gui_in[0],
            gui_in[1]),
                FONT_HERSHEY_SIMPLEX, 
                0.5, Scalar(1,1,1), 1, LINE_AA);

            imshow("output", imc);
            waitKey(1);

            // rejection criteria
            // tweak for each dataset
            if (deviation(ellipse_in, contour_x, contour_y) > d)
            {
                recenter = true;
                // skip 50 frames
                i += 49;
            }
            else
            {
                // update center and phi
                gui_in[0] = ellipse_in[0];
                gui_in[1] = ellipse_in[1];
                gui_in[5] = ellipse_in[4];
                saved_data 
                << i << ","
                << ellipse_in[0] << ","
                << ellipse_in[1] << ","
                << ellipse_in[2] << ","
                << ellipse_in[3] << ","
                << ellipse_in[4] << "\n";
            }
        }
    }
    // waitkey(0);
    destroyWindow("output");
    cout << "program finished" << endl;
    return 0;
}