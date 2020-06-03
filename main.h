#ifndef MAIN_H
#define MAIN_H

#include <QDir>
#include <QDebug>
#include <fstream>
#include <QString>
#include <iostream>
#include <QFileInfo>
#include <QStringList>
#include <QCoreApplication>

#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace face;

#define RECG_PIC_WIDTH 96
#define RECG_PIC_HEIGHT 112

void asa_clooect_face_data(string haar_face_datapath,string pic_dir_path,int device_id = 0);
void asa_produce_csv(string csv_path,string pic_path,int label);
void asa_face_recg(string haar_path,string csv_path,int recg_label,int device_id = 0);

#endif // MAIN_H
