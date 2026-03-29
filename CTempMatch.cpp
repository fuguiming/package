#include "CTempMatch.h"
#include "include_IPP/ipp.h"

//ʵ������Ƕȵ�ģ��ƥ��(ģ��ƥ��������)
//Mat& img����Դͼ��(24λ��ɫͼ��)
//Mat& temp����ģ��ͼ��(24λ��ɫͼ��)
//vector<TemplateMatchResult>& MResult����ƥ����
//����ֵ�����ش������0����û�д��� 1����Դͼ���ʽ����ȷ2����ģ��ͼ���ʽ����ȷ3����ƥ��ʧ��
int CTempMatch::TemplateMatch(Mat img, Mat temp, Rect MatchRect, float StartAngle, float EndAngle, vector<TemplateMatchResult>& MResult)
{
    // ========== ��һ������ս������������������� ==========
    MResult.clear();

    // ========== �ڶ�����У��Դͼ��Ϸ��� ==========
    if (img.empty()) {
        // ���ط�0�����룬�����ϲ��жϣ�1=Դͼ��Ϊ�գ�
        return 1;
    }
    if (img.type() != CV_8UC3)
    {
        // 2=Դͼ����24λ��ɫ
        return 2;
    }

    // ========== ��������У��ģ��ͼ��Ϸ��� ==========
    if (temp.empty()) 
    {
        // 3=ģ��ͼ��Ϊ��
        return 3;
    }
    if (temp.type() != CV_8UC3)
    {
        // 4=ģ��ͼ����24λ��ɫ
        return 4;
    }

    // ========== ���Ĳ���У��ģ��ߴ��Ƿ�Ϸ���ģ�岻�ܱ�Դͼ��� ==========
    if (temp.rows > img.rows || temp.cols > img.cols) 
    {
        // 5=ģ��ߴ糬��Դͼ��
        return 5;
    }

    // ========== ���岽������ɫͼ��ת��Ϊ�Ҷ�ͼ ==========
    cv::Mat img_gray, temp_gray;
    // COLOR_BGR2GRAY��OpenCVĬ�ϴ洢��ɫͼΪBGR��ʽ��ת��Ϊ��ͨ���Ҷ�ͼ
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(temp, temp_gray, cv::COLOR_BGR2GRAY);


    //����������ͼ��
    //ԭʼͼ����Ϊ��һ��
    //img_gray, temp_gray 
    //�ڶ���
    Mat img2, temp2;
    cv::pyrDown(img_gray, img2);
    cv::pyrDown(temp_gray, temp2);

    //������
    Mat img3, temp3;
    cv::pyrDown(img2, img3);
    cv::pyrDown(temp2, temp3);

    //���������ƥ����
    vector< TemplateMatchResult> TMResult3;
    int width = img3.cols - temp3.cols + 1;
    int height = img3.rows - temp3.rows + 1;

    //ģ�����ĵ�
    int TempCx;
    int TempCy;
    TempCx = int(temp3.cols * 0.5 + 0.5);
    TempCy = int(temp3.rows * 0.5 + 0.5);


    //ƥ������
    MatchRect.x = TempCx;
    MatchRect.y = TempCy;
    MatchRect.width = width;
    MatchRect.height = height;

    matchTemplate_ExtendAnyAngle(img3, temp3, MatchRect, 0, 360, 5, 2, 0.7, TMResult3);

    int Num = TMResult3.size();
    if (Num == 0)  //û�м�⵽
    {
        return 3;
    }
    // ���ƾ���
    for (int i = 0; i < Num; i++)
    {
        if (TMResult3[i].MaxVal > 0.5)
        {
            line(img3, TMResult3[i].CornerPt[0], TMResult3[i].CornerPt[1], Scalar(255, 0, 255), 2, 8);
            line(img3, TMResult3[i].CornerPt[1], TMResult3[i].CornerPt[2], Scalar(255, 0, 255), 2, 8);
            line(img3, TMResult3[i].CornerPt[2], TMResult3[i].CornerPt[3], Scalar(255, 0, 255), 2, 8);
            line(img3, TMResult3[i].CornerPt[3], TMResult3[i].CornerPt[0], Scalar(255, 0, 255), 2, 8);
        }
    }
    imwrite("E:/image3.bmp", img3);
    //ƥ��ڶ���
    vector< TemplateMatchResult> TMResult2;
    for (int i = 0; i < Num; i++)
    {
        TemplateMatchResult TMR = TMResult3[i];
        //ģ��ƥ������ĵ�
        //����2�ǻ��ͼ��Ŵ������������ֵ����ȥ2������2�����������5*5��Χ��ƥ��
        MatchRect.x = (TMR.maxLoc.x * 2 - 3);
        MatchRect.y = (TMR.maxLoc.y * 2 - 3);

        MatchRect.width = 7;
        MatchRect.height = 7;

        //imshow("1", img2);
        //imshow("2", temp2);
        //imwrite("E:/img2.bmp", img2);
        //imwrite("E:/temp2.bmp", temp2);
        matchTemplate_ExtendAnyAngle(img2, temp2, MatchRect, TMR.angle - 5, TMR.angle + 6, 1, 1, 0.80, TMResult2);
    }
    TMResult3.clear();

    Num = TMResult2.size();
    if (Num == 0)  //û�м�⵽
    {
        return 3;
    }
    //��ʾ�ڶ���ͼ��
    // ���ƾ���
    for (int i = 0; i < Num; i++)
    {
        if (TMResult2[i].MaxVal > 0.5)
        {
            line(img2, TMResult2[i].CornerPt[0], TMResult2[i].CornerPt[1], Scalar(255, 0, 255), 2, 8);
            line(img2, TMResult2[i].CornerPt[1], TMResult2[i].CornerPt[2], Scalar(255, 0, 255), 2, 8);
            line(img2, TMResult2[i].CornerPt[2], TMResult2[i].CornerPt[3], Scalar(255, 0, 255), 2, 8);
            line(img2, TMResult2[i].CornerPt[3], TMResult2[i].CornerPt[0], Scalar(255, 0, 255), 2, 8);
        }
    }
    imwrite("E:/image2.bmp", img2);

    //ƥ���һ��
    for (int i = 0; i < Num; i++)
    {
        TemplateMatchResult TMR = TMResult2[i];

        MatchRect.x = (TMR.maxLoc.x * 2 - 2);  //����2�ǻ��ͼ��Ŵ������������ֵ����ȥ2������2�����������5*5��Χ��ƥ��
        MatchRect.y = (TMR.maxLoc.y * 2 - 2);

        //�����Ƶ�ģ������
        //MatchRect.x += int(temp.cols * 0.5 + 0.5);
        //MatchRect.y += int(temp.rows * 0.5 + 0.5);

        MatchRect.width = 5;
        MatchRect.height = 5;


        matchTemplate_ExtendAnyAngle(img_gray, temp_gray, MatchRect, TMR.angle - 2, TMR.angle + 2, 0.2, 1, 0.90, MResult);
    }
    TMResult2.clear();

    Num = MResult.size();
    if (Num == 0)  //û�м�⵽
    {
        return 3;
    }
    // ���ƾ���
    for (int i = 0; i < Num; i++)
    {
        if (MResult[i].MaxVal > 0.5)
        {
            line(img_gray, MResult[i].CornerPt[0], MResult[i].CornerPt[1], Scalar(255, 0, 255), 2, 8);
            line(img_gray, MResult[i].CornerPt[1], MResult[i].CornerPt[2], Scalar(255, 0, 255), 2, 8);
            line(img_gray, MResult[i].CornerPt[2], MResult[i].CornerPt[3], Scalar(255, 0, 255), 2, 8);
            line(img_gray, MResult[i].CornerPt[3], MResult[i].CornerPt[0], Scalar(255, 0, 255), 2, 8);
        }
    }
    imwrite("E:/image.bmp", img_gray);

    return 0;
}
//ʵ��ͼ����չ���Ŀ����ͼ���Ե����������������Ƕȵ�ģ��ƥ��
//Rect MatchRect������ƥ������
//float StartAngle������ʼ�Ƕ�
//float EndAngle���������Ƕ�
//int cn��������
//vector<TemplateMatchResult>& MResult����ƥ����
//float Similarity�������ƶ���ֵ
void CTempMatch::matchTemplate_ExtendAnyAngle(Mat& img, Mat& temp, Rect MatchRect, float StartAngle, float EndAngle, float AngleStep, int cn, float Similarity, vector<TemplateMatchResult>& MResult)
{
    //�Ƚ���ͼ����չ��̽���Ƿ���Ҫͼ��ü���С������
    int width = img.cols - temp.cols + 1;
    int height = img.rows - temp.rows + 1;
    //ģ�����ĵ�
    int TempCx;
    int TempCy;
    TempCx = int(temp.cols * 0.5 + 0.5);
    TempCy = int(temp.rows * 0.5 + 0.5);

    int EW;
    int EH;
    int EW2;
    int EH2;
    bool IsEx = MathchTemplate_IsExtendImage(img, temp, MatchRect, EW, EH, EW2, EH2);
    //��ͼ�������չ����չ���ͼ��ߴ�
    int Height;
    int Width;
    Height = img.rows + EH + EH2;
    Width = img.cols + EW + EW2;
    Mat Image(Height, Width, CV_8UC1);
    MathchTemplate_ExtendImage(img, Image, EW, EH, EW2, EH2, 0); //��չ��������255
    //ͼ����չ��ƥ������ԭ�㣨ģ������ģ�ͬ����Ҫ�ı䣬����任����չ���ͼ������ϵ��
    MatchRect.x += EW;
    MatchRect.y += EH;



    Mat temp_R;
    Point Pt[4];

    //��ʼƥ����
    vector< TemplateMatchResult> TMResult0;

    TemplateMatchResult MaxTMR; //������ƶȣ�������е����ƶȾ�С�ڸ�������ֵ����洢������ƶ�
    MaxTMR.MaxVal = 0;


    Rect Rect0;
    Rect0 = MatchRect;


    for (float i = StartAngle; i < EndAngle; i = i + AngleStep)
    {
        //��ת�������ɵ�������Ϊ0��Pt�������ģ��ͼ�����ĵ�����ֵ
        Rotate(temp, Pt, temp_R, i, 0);

        //��ֵ��
        Mat mask(temp_R.rows, temp_R.cols, CV_8U);

        threshold(temp_R, mask, 50, 255, THRESH_BINARY);

        imwrite("E:/Mask.bmp", mask);


        //����һ������洢ƥ����
        int width = Image.cols - temp_R.cols + 1;
        int height = Image.rows - temp_R.rows + 1;

        if (width < 1 || height < 1)
        {
            continue;
        }
        Mat result(height, width, CV_32F, Scalar::all(0));


        bool IsRect;
        int N = int(i / 90);
        if (fabs(i - 90 * N) < 0.00001)
        {
            IsRect = true;
        }
        else
        {
            IsRect = false;
        }



        //��ʱ����Mask
        IsRect = true;





        //��ƥ���������ʼ�������ͼ�������ƶ�����ת��ͼ������Ͻ�
        Rect0.x = MatchRect.x - int(temp_R.cols * 0.5 + 0.5);
        Rect0.y = MatchRect.y - int(temp_R.rows * 0.5 + 0.5);
        //����ƥ��
        matchTemplate_mask(Image, temp_R, result, Rect0, cn, mask, IsRect);

        double minVal1, maxVal1;
        Point MinPt, MaxPt;

        //�õ������ֵ������ģ��ͼ��ԭ����Ŀ��ͼ���е�����ֵ
        //����Ҫ���м�������MatchRect
        int StartX, StartY, EndX, EndY;
        StartX = Rect0.x < 0 ? 0 : Rect0.x;
        StartY = Rect0.y < 0 ? 0 : Rect0.y;
        EndX = Rect0.x + Rect0.width;
        EndY = Rect0.y + Rect0.height;
        EndY = EndY < result.rows ? EndY : result.rows;
        EndX = EndX < result.cols ? EndX : result.cols;

        GetminMaxLoc(result, StartX, EndX, StartY, EndY, minVal1, maxVal1, MinPt, MaxPt);

        while (maxVal1 > Similarity)
        {
            TemplateMatchResult TMR;
            TMR.angle = i;
            TMR.MaxVal = maxVal1;
            //ģ��ͼ��ԭ����Ŀ��ͼ���е�����ֵ
            //TMR.maxLoc = MaxPt; 
            //ģ��ͼ��������Ŀ��ͼ���е�����ֵ
            TMR.maxLoc.x = MaxPt.x + int(temp_R.cols * 0.5 + 0.5);
            TMR.maxLoc.y = MaxPt.y + int(temp_R.rows * 0.5 + 0.5);

            TMR.CornerPt[0].x = Pt[0].x + TMR.maxLoc.x;
            TMR.CornerPt[0].y = Pt[0].y + TMR.maxLoc.y;

            TMR.CornerPt[1].x = Pt[1].x + TMR.maxLoc.x;
            TMR.CornerPt[1].y = Pt[1].y + TMR.maxLoc.y;

            TMR.CornerPt[2].x = Pt[2].x + TMR.maxLoc.x;
            TMR.CornerPt[2].y = Pt[2].y + TMR.maxLoc.y;

            TMR.CornerPt[3].x = Pt[3].x + TMR.maxLoc.x;
            TMR.CornerPt[3].y = Pt[3].y + TMR.maxLoc.y;

            TMResult0.push_back(TMR);

            GetNextMaxLoc(result, StartX, EndX, StartY, EndY, MaxPt, temp_R.cols, temp_R.rows, maxVal1, MaxPt);
        }

        //  if (maxVal1 < Similarity) //��ȡ������ƶ�
        if (TMResult0.size() == 0)
        {
            if (maxVal1 > MaxTMR.MaxVal)
            {
                MaxTMR.angle = i;
                MaxTMR.MaxVal = maxVal1;
                //
               // MaxTMR.maxLoc = MaxPt;
                //ģ��ͼ��������Ŀ��ͼ���е�����ֵ
                MaxTMR.maxLoc.x = MaxPt.x + int(temp_R.cols * 0.5 + 0.5);
                MaxTMR.maxLoc.y = MaxPt.y + int(temp_R.rows * 0.5 + 0.5);

                MaxTMR.CornerPt[0].x = Pt[0].x + MaxTMR.maxLoc.x;
                MaxTMR.CornerPt[0].y = Pt[0].y + MaxTMR.maxLoc.y;

                MaxTMR.CornerPt[1].x = Pt[1].x + MaxTMR.maxLoc.x;
                MaxTMR.CornerPt[1].y = Pt[1].y + MaxTMR.maxLoc.y;

                MaxTMR.CornerPt[2].x = Pt[2].x + MaxTMR.maxLoc.x;
                MaxTMR.CornerPt[2].y = Pt[2].y + MaxTMR.maxLoc.y;

                MaxTMR.CornerPt[3].x = Pt[3].x + MaxTMR.maxLoc.x;
                MaxTMR.CornerPt[3].y = Pt[3].y + MaxTMR.maxLoc.y;
            }
        }

    }




    TemplateMatchResult MaxTMR0;
    TemplateMatchResult MaxTMR1;//����任
    GetMaxMatchResult(TMResult0, MaxTMR0);

    while (MaxTMR0.MaxVal > Similarity)
    {
        //�����Ŀ��ͼ���������չ����Ҫ������ֵ�仯��ԭĿ��ͼ������ϵ��
        if (IsEx)
        {

            MaxTMR1 = MaxTMR0;

            MaxTMR1.maxLoc.x -= EW;
            MaxTMR1.maxLoc.y -= EH;
            for (int m = 0; m < 4; m++)
            {
                MaxTMR1.CornerPt[m].x -= EW;
                MaxTMR1.CornerPt[m].y -= EH;
            }
        }
        else
        {
            MaxTMR1 = MaxTMR0;
        }
        MResult.push_back(MaxTMR1);
        GetNextMaxMatchResult(TMResult0, MaxTMR0, temp.cols, temp.rows);
    }
    TMResult0.clear();
    if (MResult.size() == 0 && MaxTMR.MaxVal > 0) //û�б�Similarity�����洢����
    {
        if (IsEx)
        {
            MaxTMR.maxLoc.x -= EW;
            MaxTMR.maxLoc.y -= EH;
            for (int m = 0; m < 4; m++)
            {
                MaxTMR.CornerPt[m].x -= EW;
                MaxTMR.CornerPt[m].y -= EH;
            }
        }
        MResult.push_back(MaxTMR);
    }
}

bool CTempMatch::MathchTemplate_IsExtendImage(Mat Src, Mat temp, Rect MatchRect, int& EW, int& EH, int& EW2, int& EH2)
{
    //ģ��Խǳߴ��һ��
    EW = 0;
    EH = 0;
    EW2 = 0;
    EH2 = 0;
    //б�ǰ�߳��ȵ�һ��
    int L = int(sqrt(temp.cols * temp.cols + temp.rows * temp.rows) * 0.5 + 0.5);
    if (MatchRect.x < L || MatchRect.y < L || (MatchRect.x + MatchRect.width) + L>Src.cols || (MatchRect.y + MatchRect.height) + L>Src.rows) //�Ƿ�Ҫ��ͼ�������չ
    {
        //Ҫ������չ�ĳߴ�
        EW = L > MatchRect.x ? L - MatchRect.x : 0;
        EH = L > MatchRect.y ? L - MatchRect.y : 0;
        EW2 = (MatchRect.x + MatchRect.width) + L > Src.cols ? (MatchRect.x + MatchRect.width) + L - Src.cols : 0;
        EH2 = (MatchRect.y + MatchRect.height) + L > Src.rows ? (MatchRect.y + MatchRect.height) + L - Src.rows : 0;

        return true;
    }
    else
    {
        return false;
    }
}

// ģ��ƥ���ж�Ŀ��ͼ�����˫����չ�������ҷֱ���չEW�����·ֱ���չEH
// Mat Src����Ŀ��ͼ��
// Mat& Dst������չ���ͼ��
// int EW����Src�����չ�Ŀ���
// int EH����Src������չ�ĸ߶�
// int EW2����Src�Ҳ���չ�Ŀ���
// int EH2����Src�ײ���չ�ĸ߶�
void CTempMatch::MathchTemplate_ExtendImage(Mat Src, Mat& Dst, int EW, int EH, int EW2, int EH2, int Value)
{
    if (EW == 0 && EH == 0 && EW2 == 0 && EH2 == 0)
    {
        Dst = Src.clone();
        return;
    }


    //������ͼ��
    for (int i = 0; i < Dst.rows; i++)
    {
        for (int j = 0; j < Dst.cols; j++)
        {
            if (i < EH || i >= Src.rows + EH || j < EW || j >= Src.cols + EW)
            {
                Dst.at<uchar>(i, j) = Value;
            }
            else
            {
                Dst.at<uchar>(i, j) = Src.at<uchar>(i - EH, j - EW);
            }
        }
    }
}

//ͼ����ת
void CTempMatch::Rotate(const Mat& SrcImage, Point Pt[4], Mat& DestImage, double angle, int BValue)
{
    int W = SrcImage.cols;
    int H = SrcImage.rows;

    //ͼ���ĸ��ǵ�����
    Pt[0].x = 0;
    Pt[0].y = 0;

    Pt[1].x = 0;
    Pt[1].y = H - 1;

    Pt[2].x = W - 1;
    Pt[2].y = H - 1;

    Pt[3].x = W - 1;
    Pt[3].y = 0;

    int Cx, Cy;
    Cx = int(W * 0.5 + 0.5);
    Cy = int(H * 0.5 + 0.5);
    Point2f center(Cx, Cy); //����

    //������ת�ķ���任����
    Mat M = getRotationMatrix2D(center, angle, 1);

    //��ת��ͼ��Ŀ���ֵ�͸߶�ֵ
    double cos1 = abs(M.at<double>(0, 0));
    double sin1 = abs(M.at<double>(0, 1));
    int NW = (int)(cos1 * W + sin1 * H + 0.5);
    int NH = (int)(sin1 * W + cos1 * H + 0.5);

    M.at<double>(0, 2) += (NW * 0.5 - W * 0.5 + 0.5);//�������ĵ�ˮƽƽ�Ƶľ��룬����������ƽ��
    M.at<double>(1, 2) += (NH * 0.5 - H * 0.5 + 0.5);//�������ĵ���ֱƽ�Ƶľ��룬����������ƽ��

    //��ͼ���ĸ��ǵ������ת
    for (int i = 0; i < 4; i++)
    {
        //���ĸ��ǵ�ת������������ϵ,��������ԭ��任��ͼ������
        Pt[i].x = Pt[i].x - center.x;
        Pt[i].y = -Pt[i].y + center.y;

        float Sita = angle * 3.1415926 / 180;
        float C = cos(Sita);
        float S = sin(Sita);

        int X, Y;

        X = int(C * Pt[i].x - S * Pt[i].y + 0.5);
        Y = int(S * Pt[i].x + C * Pt[i].y + 0.5);

        //��������ԭ��ת������ͼ������Ͻǣ���ת��Ϊ��������ϵ
      //  Pt[i].x = X + NW;
      //  Pt[i].y = -(Y - NH / 2);

        //�����ͼ�����ĵ�����ֵ
        Pt[i].x = X;
        Pt[i].y = -Y;
    }
     warpAffine(SrcImage, DestImage, M, Size(NW, NH),1,0,255); //����任��������������Ϊ255
    //warpAffine(SrcImage, DestImage, M, Size(NW, NH));
     warpAffine(SrcImage, DestImage, M, Size(NW, NH), 1, 0, BValue); //����任��������������Ϊ255
}

//Rect MatchRect����Ҫ����ƥ�������Χ
//int cn��������
void CTempMatch::matchTemplate_mask(Mat& img, Mat& templ, Mat& result, Rect MatchRect, int cn, Mat& mask, bool IsRect)
{
    //���mask��ÿһ�еı߽�㣨�������а�ɫ�㣩
    vector<MaskContour> MaskContPts;
    int Num; //ԭʼģ������ص�����Ҳ�������а׵�����ص���
    double templSum, templSq; //����Լ�ƽ����

    if (IsRect == true) //��������
    {
        Num = templ.rows * templ.cols;
        ComputeSumSq(templ, templSum, templSq);
    }
    else  //�ж϶���ģ��ͼ����Ҫ�õ�maskͼ�����������/���е����ص�����ֵ
    {
        if (templ.rows < templ.cols)  //��>��
        {
            DetectMaskContour_X(templ, mask, MaskContPts, Num, templSum, templSq);
        }
        else
        {
            DetectMaskContour_Y(templ, mask, MaskContPts, Num, templSum, templSq);
        }

    }




    if (Num == 0)
    {
        return;
    }
    //��ע����1������ģ����ת������б���Σ������ʹ�û���ͼ����ʱ�����м��㣻
    //      ��2����ʽ��ģ��Ŀ��͸ߵĳ˻���W*H�����ڴ���ģ��ʱ��ͳ��ģ���а�ɫ��������ص�������
    IppiSize srcRoiSize = { img.cols,img.rows };
    IppiSize tplRoiSize = { templ.cols,templ.rows };

    //   IppAutoBuffer<Ipp8u> buffer;
    int bufSize = 0;
    IppEnum funCfg = (IppEnum)(ippAlgAuto | ippiROIValid);

    IppStatus status = ippiCrossCorrNormGetBufferSize(srcRoiSize, tplRoiSize, funCfg, &bufSize);

    Ipp8u* buffer;

    buffer = ippsMalloc_8u(bufSize);

    //����Ŀ��ͼ���ģ��ͼ��ľ��� ��  ��ʽ�е�1�� �� ���洢��result��
    ippiCrossCorrNorm_8u32f_C1R(img.ptr(), (int)img.step, srcRoiSize, templ.ptr(), (int)templ.step, tplRoiSize, result.ptr<Ipp32f>(), (int)result.step, funCfg, buffer);

    ippsFree(buffer);

    //���㹫ʽ�е�4��͵�5��
    double invArea = 1. / ((double)Num);
    double R45 = sqrt(templSq - templSum * templSum * invArea);

    //�������ͼ
    Mat sum, sqsum;
    integral(img, sum, sqsum, CV_64F);

    double* q0 = 0, * q2 = 0;
    int sumstep = 0;

    CV_Assert(sqsum.data != NULL);
    q0 = (double*)sqsum.data;
    q2 = (double*)(sqsum.data + sqsum.step);//1��


    CV_Assert(sum.data != NULL);
    double* p0 = (double*)sum.data;
    double* p2 = (double*)(sum.data + sum.step);  //1��


    sumstep = sum.data ? (int)(sum.step / sizeof(double)) : 0;

    int i, j, k;
    double SrcSum, SrcSq;

    float* rrow;
    int idx;
    //����Ҫ���м�������MatchRect
    int StartX, StartY, EndX, EndY, Width, Height;
    StartX = MatchRect.x < 0 ? 0 : MatchRect.x;
    StartY = MatchRect.y < 0 ? 0 : MatchRect.y;
    Width = MatchRect.width;
    Height = MatchRect.height;
    EndX = MatchRect.x + MatchRect.width;
    EndY = MatchRect.y + MatchRect.height;
    EndY = EndY < result.rows ? EndY : result.rows;
    EndX = EndX < result.cols ? EndX : result.cols;

    for (i = StartY; i < EndY; i += cn)
    {
        rrow = result.ptr<float>(i);
        idx = i * sumstep;

        for (j = StartX, idx += StartX; j < EndX; j += cn, idx += cn)
        {
            if (i == 8 && j == 23)
            {
                int MM = 0;
            }
            double num = rrow[j], t, t1, t2;
            //�ɻ���ͼ����Ŀ��ͼ��ĺ�SrcSum�Լ�ƽ����SrcSq;
            SrcSum = 0;
            SrcSq = 0;
            if (IsRect == true) //����
            {
                t1 = p0[idx] - p0[idx + templ.cols] - p0[idx + templ.rows * sumstep] + p0[idx + templ.rows * sumstep + templ.cols];
                SrcSum += t1;

                t2 = q0[idx] - q0[idx + templ.cols] - q0[idx + templ.rows * sumstep] + q0[idx + templ.rows * sumstep + templ.cols];
                SrcSq += t2;
            }
            else
            {
                if (templ.rows < templ.cols)  //��>��
                {
                    for (k = 0; k < MaskContPts.size(); k++)
                    {
                        Point LeftPt, RightPt;
                        LeftPt = MaskContPts[k].LeftPt;
                        RightPt = MaskContPts[k].RightPt;
                        int y = LeftPt.y * sumstep + idx;

                        t1 = p0[y + LeftPt.x] - p0[y + RightPt.x] - p2[y + LeftPt.x] + p2[y + RightPt.x];
                        SrcSum += t1;

                        t2 = q0[y + LeftPt.x] - q0[y + RightPt.x] - q2[y + LeftPt.x] + q2[y + RightPt.x];
                        SrcSq += t2;
                    }
                }
                else
                {
                    for (k = 0; k < MaskContPts.size(); k++)
                    {
                        Point UpPt, DownPt;
                        UpPt = MaskContPts[k].LeftPt;
                        DownPt = MaskContPts[k].RightPt;

                        t1 = p0[idx + (UpPt.y) * sumstep + UpPt.x] - p0[idx + (UpPt.y) * sumstep + UpPt.x + 1] - p0[idx + DownPt.y * sumstep + DownPt.x] + p0[idx + DownPt.y * sumstep + DownPt.x + 1];
                        SrcSum += t1;

                        t2 = q0[idx + (UpPt.y) * sumstep + UpPt.x] - q0[idx + (UpPt.y) * sumstep + UpPt.x + 1] - q0[idx + DownPt.y * sumstep + DownPt.x] + q0[idx + DownPt.y * sumstep + DownPt.x + 1];
                        SrcSq += t2;
                    }
                }

            }


            //��ʽ��6��͵�7��
            double diff2 = MAX(SrcSq - SrcSum * SrcSum * invArea, 0);

            t = std::sqrt(diff2) * R45;

            rrow[j] = (float)(num - templSum * SrcSum * invArea) / t;
        }
    }
}

//ֻ���ƥ�������Ҫ��Ȼ��������Ŀ��
void CTempMatch::GetminMaxLoc(const Mat& src, int StartX, int EndX, int StartY, int EndY, double& minVal, double& maxVal, Point& MinPt, Point& MaxPt)
{

    maxVal = 0;
    minVal = 3.4E+38;

    float Value;
    for (int y = StartY; y < EndY; y++)
    {
        for (int x = StartX; x < EndX; x++)
        {
            if (x == 23 && y == 8)
            {
                int M = 0;
            }
            Value = src.at<float>(y, x);
            if (Value < -1 || Value>1)
            {
                int s = 0;
            }
            else
            {
                if (minVal > Value)
                {
                    minVal = Value;
                    MinPt.x = x;
                    MinPt.y = y;
                }
                if (maxVal < Value)
                {

                    maxVal = Value;
                    MaxPt.x = x;
                    MaxPt.y = y;
                }
            }

        }
    }
}

//�����һ������
void CTempMatch::GetNextMaxLoc(Mat& result, int StartX, int EndX, int StartY, int EndY, Point maxLoc, int templatW, int templatH, double& NewMaxValue, Point& NewMaxLoc)
{
    NewMaxValue = 0;

    int startX = maxLoc.x - templatW / 3;
    int startY = maxLoc.y - templatH / 3;
    int endX = maxLoc.x + templatW / 3;
    int endY = maxLoc.y + templatH / 3;
    if (startX < 0)
    {
        startX = 0;
    }
    if (startY < 0)
    {
        startY = 0;
    }
    if (endX > result.cols)
    {
        endX = result.cols;

    }
    if (endY > result.rows)
    {
        endY = result.rows;
    }
    int y, x;
    for (y = startY; y < endY; y++)
    {
        float* data = result.ptr<float>(y);

        for (x = startX; x < endX; x++)
        {
            data[x] = 0;
        }
    }

    double NewminVal;
    Point  MinPt;
    GetminMaxLoc(result, StartX, EndX, StartY, EndY, NewminVal, NewMaxValue, MinPt, NewMaxLoc);
}

void CTempMatch::GetMaxMatchResult(vector<TemplateMatchResult> TMResult, TemplateMatchResult& MaxTMR)
{
    int Num = TMResult.size();
    MaxTMR.MaxVal = 0;
    if (Num == 0)
    {
        return;
    }
    MaxTMR = TMResult[0];

    for (int i = 1; i < Num; i++)
    {

        if (TMResult[i].MaxVal > MaxTMR.MaxVal)
        {
            MaxTMR = TMResult[i];
        }
    }
}

//TMResult�п��ܴ������ڽǶ��µĶ����ֵ�㣬�Ƚ����ֵ�ռ䷶Χ�ڵļ�ֵ�����������Ȼ���ٲ�����һ������
void CTempMatch::GetNextMaxMatchResult(vector<TemplateMatchResult>& TMResult, TemplateMatchResult& MaxTMR, int templatW, int templatH)
{
    MaxTMR.MaxVal = 0;

    int Num = TMResult.size();
    if (Num == 0)
    {
        return;
    }

    int startX = MaxTMR.maxLoc.x - templatW / 3;
    int startY = MaxTMR.maxLoc.y - templatH / 3;
    int endX = MaxTMR.maxLoc.x + templatW / 3;
    int endY = MaxTMR.maxLoc.y + templatH / 3;


    int x, y;

    vector<TemplateMatchResult>::iterator itor;
    for (itor = TMResult.begin(); itor != TMResult.end(); )
    {

        x = itor->maxLoc.x;
        y = itor->maxLoc.y;

        if (x > startX && x<endX && y>startY && y < endY)
        {
            //   TMResult.swap(itor, TMResult.back());
            //   TMResult.swap(itor);

            itor = TMResult.erase(itor);
            //  itor--;
        }
        else
        {
            itor++;
        }
    }

    Num = TMResult.size();

    if (Num == 0)
    {
        return;
    }

    MaxTMR = TMResult[0];


    for (int i = 1; i < Num; i++)
    {

        if (TMResult[i].MaxVal > MaxTMR.MaxVal)
        {
            MaxTMR = TMResult[i];
        }
    }
}

//����ͼ�����صĺ��Լ�ƽ����
void CTempMatch::ComputeSumSq(Mat& temp, double& sum, double& sqsum)
{
    sum = 0;
    sqsum = 0;

    for (int i = 0; i < temp.rows; i++)
    {
        uchar* temprrow = temp.ptr<uchar>(i);

        for (int j = 0; j < temp.cols; j++)
        {
            sum += temprrow[j];
            sqsum += temprrow[j] * temprrow[j];
        }
    }
}

void CTempMatch::DetectMaskContour_X(Mat& temp, Mat& mask, vector<MaskContour>& MaskContPts, int& Num, double& sum, double& sqsum)
{
    int W = mask.cols;
    int H = mask.rows;

    bool IsDet;
    Num = 0;
    sum = 0;
    sqsum = 0;

    int Num2 = 0;

    for (int i = 0; i < H; i++)
    {
        uchar* temprrow = temp.ptr<uchar>(i);
        uchar* maskrrow = mask.ptr<uchar>(i);

        IsDet = false;

        MaskContour MaskPt;
        MaskPt.LeftPt.x = W;
        MaskPt.LeftPt.y = i;
        MaskPt.RightPt.x = 0;
        MaskPt.RightPt.y = i;

        for (int j = 0; j < W; j++)
        {
            if (maskrrow[j] == 255) //Ŀ��Ϊ��ɫ
            {
                if (MaskPt.LeftPt.x > j)
                {
                    MaskPt.LeftPt.x = j;
                }
                if (MaskPt.RightPt.x < j)
                {
                    MaskPt.RightPt.x = j;
                }

                IsDet = true;
                //  Num++;
                //  sum += temprrow[j];
                //  sqsum += temprrow[j] * temprrow [j];
            }
        }

        if (IsDet)
        {
            Num += MaskPt.RightPt.x - MaskPt.LeftPt.x + 1;
            //����MaskPt.LeftPt.x��MaskPt.RightPt.x��ֵ
            for (int x = MaskPt.LeftPt.x; x < MaskPt.RightPt.x + 1; x++)
            {
                sum += temprrow[x];
                sqsum += temprrow[x] * temprrow[x];
            }
            MaskContPts.push_back(MaskPt);
        }
    }
}

//��Y����ɼ���Ե��
void CTempMatch::DetectMaskContour_Y(Mat& temp, Mat& mask, vector<MaskContour>& MaskContPts, int& Num, double& sum, double& sqsum)
{
    int W = mask.cols;
    int H = mask.rows;

    bool IsDet;
    Num = 0;
    sum = 0;
    sqsum = 0;

    for (int j = 0; j < W; j++)
    {
        IsDet = false;

        MaskContour MaskPt;
        MaskPt.LeftPt.x = j;    //�ϵ�
        MaskPt.LeftPt.y = H;
        MaskPt.RightPt.x = j;   //�µ�
        MaskPt.RightPt.y = 0;

        for (int i = 0; i < H; i++)
        {
            uchar tempPixel = temp.at<uchar>(i, j);
            uchar maskPixel = mask.at<uchar>(i, j);

            if (maskPixel == 255)
            {
                if (MaskPt.LeftPt.y > i)
                {
                    MaskPt.LeftPt.y = i;
                }
                if (MaskPt.RightPt.y < i)
                {
                    MaskPt.RightPt.y = i;
                }

                IsDet = true;
                // Num++;
                // sum += tempPixel;
                // sqsum += tempPixel * tempPixel;
            }
        }
        if (IsDet)
        {
            Num += MaskPt.RightPt.x - MaskPt.LeftPt.x + 1;
            //����MaskPt.LeftPt.x��MaskPt.RightPt.x��ֵ
            for (int y = MaskPt.LeftPt.x; y < MaskPt.RightPt.x + 1; y++)
            {
                uchar tempPixel = temp.at<uchar>(y, j);
                sum += tempPixel;
                sqsum += tempPixel * tempPixel;
            }
            MaskContPts.push_back(MaskPt);
        }
    }
}


//ͼ����
int CTempMatch::MSV_TempImageDifference(Mat img, Mat temp, float angle, Point maxLocPt, int Threshold, Mat& cvDImg)
{
    //LPSTR lpDIBBits, LONG lWidth, LONG lHeight, LPSTR lpTempDIBBits, LONG lTempWidth, LONG lTempHeight
    //��ģ��ͼ��ת��ΪOpenCVͼ��
    //Mat temp(lTempHeight, lTempWidth, CV_8U);
    //DIBBitsToCVImage(lpTempDIBBits, lTempWidth, lTempHeight, temp);

    //��Դͼ��ת��ΪOpenCV��ʽ
    //Mat cvSrcImg(lHeight, lWidth, CV_8U);
    //DIBBitsToCVImage(lpDIBBits, lWidth, lHeight, cvSrcImg);
     // ========== �ڶ�����У��Դͼ��Ϸ��� ==========
    if (img.empty()) {
        // ���ط�0�����룬�����ϲ��жϣ�1=Դͼ��Ϊ�գ�
        return 1;
    }
    if (img.type() != CV_8UC3)
    {
        // 2=Դͼ����24λ��ɫ
        return 2;
    }

    // ========== ��������У��ģ��ͼ��Ϸ��� ==========
    if (temp.empty())
    {
        // 3=ģ��ͼ��Ϊ��
        return 3;
    }
    if (temp.type() != CV_8UC3)
    {
        // 4=ģ��ͼ����24λ��ɫ
        return 4;
    }

    // ========== ���Ĳ���У��ģ��ߴ��Ƿ�Ϸ���ģ�岻�ܱ�Դͼ��� ==========
    if (temp.rows > img.rows || temp.cols > img.cols)
    {
        // 5=ģ��ߴ糬��Դͼ��
        return 5;
    }

    // ========== ���岽������ɫͼ��ת��Ϊ�Ҷ�ͼ ==========
    cv::Mat img_gray, temp_gray;
    // COLOR_BGR2GRAY��OpenCVĬ�ϴ洢��ɫͼΪBGR��ʽ��ת��Ϊ��ͨ���Ҷ�ͼ
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(temp, temp_gray, cv::COLOR_BGR2GRAY);
    //������ת
    Mat temp_R;
    Point Pt[4];
    Rotate(temp_gray, Pt, temp_R, angle, 0);

    int TempH, TempW;
    TempH = temp_R.rows;
    TempW = temp_R.cols;
    int StartX, StartY;
    int TempCx, TempCy;
    //  TempCx = int(float(TempW) / 2 + 0.5);
    //  TempCy = int(float(TempH) / 2 + 0.5);
    TempCx = int(float(TempW) / 2);
    TempCy = int(float(TempH) / 2);
    StartX = maxLocPt.x - TempCx; //��ƥ�����ĵ�maxLocPtƽ�Ƶ����Ͻ�
    StartY = maxLocPt.y - TempCy;

    for (int i = 1; i < TempH - 1; i++)  //������������ȡ����ɵĵ�����ƫ���������1������
    {
        for (int j = 1; j < TempW - 1; j++)
        {
            uchar TempGray = temp_R.at<uchar>(i, j);
            if ((i + StartY) >= 0 && (i + StartY) < img_gray.rows && (j + StartX) >= 0 && (j + StartX) < img_gray.cols)
            {
                uchar SrcGray = img_gray.at<uchar>(i + StartY, j + StartX);
                if (abs(SrcGray - TempGray) > Threshold)
                {
                    cvDImg.at<uchar>(i + StartY, j + StartX) = 255;
                }
                else
                {
                    cvDImg.at<uchar>(i + StartY, j + StartX) = 0;
                }

            }

        }
    }

    return 0;
}
