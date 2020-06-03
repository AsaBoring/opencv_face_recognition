#include "main.h"

void asa_clooect_face_data(string haar_face_datapath, string pic_dir_path, int device_id)
{
    /*open video capture*/
    VideoCapture capture(device_id);
    if(!capture.isOpened())
    {
        cout<<"can't open camera";
        return;
    }

    /*load haar face model*/
    CascadeClassifier haar_cascade;
    haar_cascade.load(haar_face_datapath);
    Mat frame;
    vector<Rect> faces;
    size_t count = 0;
    while(capture.read((frame)))
    {
        flip(frame,frame,1);
        haar_cascade.detectMultiScale(frame,faces);
        for(size_t i = 0 ; i < faces.size() ; ++i)
        {
            /*rectangle face rect on video*/
            rectangle(frame,faces[i],Scalar(0,0,255),2,8,0);
            if(count % 10 == 0)
            {
                Mat dst;
                resize(frame(faces[i]),dst,Size(RECG_PIC_WIDTH,RECG_PIC_HEIGHT));
                char str_count[128];
                snprintf(str_count,count,"%d");
                string pic_path = pic_dir_path + "face_" + string(str_count) + ".png";

                /*save recg face pic*/
                imwrite(pic_path.c_str(),dst);
            }
        }
        imshow("capture",frame);
        char c = waitKey(10);
        if(c == 27)
        {
            break;
        }
        count++;
    }

    capture.release();
}

void asa_produce_csv(string csv_path, string pic_path, int label)
{
    FILE * fp = fopen(csv_path.data(),"w");
    if(!fp)
    {
        cout<<"create csv file failed";
        return;
    }

    QDir image_dir(pic_path.data());
    if(image_dir.isEmpty())
    {
        cout<<"pic dir path is null"<<endl;
        return;
    }
    image_dir.setFilter(QDir::Files);
    image_dir.setSorting(QDir::Name);
    image_dir.setNameFilters(QStringList("*.png"));
    QFileInfoList file_name_list = image_dir.entryInfoList();
    for(int i = 0 ; i < file_name_list.size() ; ++i)
    {
        cout<<i<<" path is :"<<file_name_list.at(i).absoluteFilePath().toUtf8().constData()<<endl;
        QString tmp_str = file_name_list.at(i).absoluteFilePath() + ";" + label + "\n";
        cout<<"tmp_str is :"<<tmp_str.toUtf8().constData()<<endl;
        fwrite(tmp_str.toUtf8().constData(),tmp_str.size(),1,fp);
    }
    fclose(fp);
}

void asa_face_recg(string haar_path, string csv_path, int recg_label, int device_id)
{

    vector<Mat> images;
    vector<int> labels;

    /*load recg data by csv*/
    ifstream file(csv_path,ifstream::in);
    if(!file)
    {
        cout<<"load recg data faield"<<endl;
        return;
    }
    string line,path,label;
    while (getline(file,line)) {
        stringstream liness(line);
        getline(liness,path,';');
        getline(liness,label);
        if(!path.empty() && !label.empty())
        {
            images.push_back(imread(path,0));
            labels.push_back(atoi(label.c_str()));
        }
    }

    /*training model*/
    Ptr<EigenFaceRecognizer> recg_model = EigenFaceRecognizer::create();
    recg_model->train(images,labels);
    recg_model->save("./eigenface.yml");

    /*load haar model*/
    CascadeClassifier haar_cascade;
    haar_cascade.load(haar_path);

    /*start up capture*/
    VideoCapture cap(device_id);
    if(!cap.isOpened())
    {
        cout<<"open video capture failed"<<endl;
        return;
    }
    Mat frame;
    int width = images[0].cols;
    int height = images[0].rows;
    while (cap.read(frame)) {
        /*旋转成像*/
        flip(frame,frame,1);
        Mat gray;
        cvtColor(frame,gray,COLOR_BGR2GRAY);
        vector<Rect> faces;
        haar_cascade.detectMultiScale(gray,faces);
        for(size_t i = 0 ; i < faces.size() ; ++i)
        {
            Rect face_i = faces[i];
            Mat face = gray(face_i);
            Mat predict_face;
            resize(face,predict_face,Size(width,height),1.0,1.0,INTER_CUBIC);
            int face_label = recg_model->predict(predict_face);
            rectangle(frame,faces[i],Scalar(0,0,255),2,8,0);
            putText(frame,face_label==recg_label?"This is Asa , hello":"who are tou",face_i.tl(),FONT_HERSHEY_PLAIN,1.0,Scalar(255,0,0));
        }

        imshow("recg_frame",frame);
        char c = (char)waitKey(10);
        if(c == 27)
        {
            break;
        }
    }
}

void asa_face_recg_by_load_yml(string haar_path , string yml_path, int recg_label, int device_id)
{
    /*load yml file*/
    Ptr<EigenFaceRecognizer> recg_model = Algorithm::load<EigenFaceRecognizer>(yml_path.c_str());

    /*load haar model*/
    CascadeClassifier haar_cascade;
    haar_cascade.load(haar_path);

    /*start up capture*/
    VideoCapture cap(device_id);
    if(!cap.isOpened())
    {
        cout<<"open video capture failed"<<endl;
        return;
    }
    Mat frame;
    int width = 96;
    int height = 112;
    while (cap.read(frame)) {
        /*旋转成像*/
        flip(frame,frame,1);
        Mat gray;
        cvtColor(frame,gray,COLOR_BGR2GRAY);
        vector<Rect> faces;
        haar_cascade.detectMultiScale(gray,faces);
        for(size_t i = 0 ; i < faces.size() ; ++i)
        {
            Rect face_i = faces[i];
            Mat face = gray(face_i);
            Mat predict_face;
            resize(face,predict_face,Size(width,height),1.0,1.0,INTER_CUBIC);
            int face_label = recg_model->predict(predict_face);
            rectangle(frame,faces[i],Scalar(0,0,255),2,8,0);
            putText(frame,face_label==recg_label?"This is Asa , hello":"who are tou",face_i.tl(),FONT_HERSHEY_PLAIN,1.0,Scalar(255,0,0));
        }

        imshow("recg_frame",frame);
        char c = (char)waitKey(10);
        if(c == 27)
        {
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    /*
     * 此项目运行需要添加opencv+contrib的lib
     * 先执行asa_clooect_face_data收集面部数据
     * 在执行asa_produce_csv生成csv文件以便face recg模型使用
     * 最后执行asa_face_recg保存模型文件查看识别效果如何
     * 另外asa_face_recg_by_load_yml加载b本地的yml文件进行识别
    */

    /*面部识别的图像标记，可自定义*/
    int recg_label = 19;

    /*存放面部图片的文件夹路径*/
    string dir_path = "C:/Users/houxia2x/Desktop/Asa/opencv/project/PIC_DATA/";
    /*csv文件的绝对路径*/
    string csv_path = "C:/Users/houxia2x/Desktop/Asa/opencv/project/PIC_DATA/image.csv";
    /*opencv的haar model的文件路径*/
    string haar_path = "C:/Users/houxia2x/Desktop/Asa/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_alt_tree.xml";
    /*yml文件路径*/
    string yml_path = "./eigenfaces.yml";
    //    /*收集人脸图片
    //     * 1.在执行此函数时请头部保持不动，在眉眼保持不变的情况下做尽可能多的表情
    //     * 2.此函数每10帧进行一次图片保存，大概保存15~20张图片即可，请自行按ESC键退出
    //     * 3.在图片生成完毕即此函数结束后，请在生成的图片中选择眉眼位置成一条线的不同表情的图片15~20张并为其重新顺序命名
    //    */
    //    asa_clooect_face_data(haar_path,dir_path);

    /*将上一步骤生成的人脸图片以及对应的label写入csv文件*/
    //    asa_produce_csv(csv_path,dir_path);

    /*上述两步完成后即可执行此函数进行是被效果验证*/
    //    asa_face_recg(haar_path,csv_path,recg_label);

    /*加载本地识别模型进行人脸识别*/
    asa_face_recg_by_load_yml(haar_path,yml_path,recg_label);
    return a.exec();
}
