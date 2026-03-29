#pragma once
//ģ��ƥ����
#include <opencv2/opencv.hpp> // OpenCV����ͷ�ļ�
#include <iostream>

using namespace cv;
using namespace std;

struct TemplateMatchResult
{
	float MaxVal;    //���ƶ����ֵ     
	float angle;     //���ƶ����ֵʱ�ĽǶ�
	Point maxLoc;    //���ƶ����ֵʱ��λ��(ģ��ͼ�����ģ�
	Point CornerPt[4];//���ƶ����ֵʱ���ĸ��ǵ�����
};
struct MaskContour //mask�������������¼ÿһ�е����Ұ�ɫ��
{
	Point LeftPt;
	Point RightPt;
};
class CTempMatch
{
public:
	//ʵ������Ƕȵ�ģ��ƥ��
	//Mat& img����Դͼ��(24λ��ɫͼ��)
	//Mat& temp����ģ��ͼ��(24λ��ɫͼ��)
	//Rect MatchRect������ƥ������
	//float StartAngle������ʼ�Ƕ�
	//float EndAngle���������Ƕ�
	//vector<TemplateMatchResult>& MResult����ƥ����
	//����ֵ�����ش������0����û�д��� 1����Դͼ���ʽ����ȷ2����ģ��ͼ���ʽ����ȷ3����ƥ��ʧ��
	int TemplateMatch(Mat img, Mat temp, Rect MatchRect, float StartAngle, float EndAngle,vector<TemplateMatchResult>& MResult);

	//ģ��ͼ�������ͼ���֣�������ߵĲ���
	//LPSTR lpDIBBits��������ͼ������
	//LONG lWidth��������ͼ�����
	//LONG lHeight��������ͼ��߶�
	//LPSTR lpTempDIBBits����ģ��ͼ��
	//LONG lTempWidth����ģ��ͼ�����
	//LONG lTempHeight����ģ��ͼ��߶�
	//float angle������ת�Ƕ�
	//POINT maxLocPt����ƥ��λ������
	//int Threshold���������ֵ������Threshold��Ϊ255��������Ϊ0
	//Mat& cvDImg������ֺ�ͼ�����ݣ���СΪ��lWidth,lHeight��
	//����ֵ�����ش������0����û�д��� 1����Դͼ���ʽ����ȷ2����ģ��ͼ���ʽ����ȷ
	int MSV_TempImageDifference(Mat img, Mat temp, float angle, Point maxLocPt, int Threshold, Mat& cvDImg);
private:
	//ʵ��ͼ����չ���Ŀ����ͼ���Ե����������������Ƕȵ�ģ��ƥ��
	//Mat& img����Դͼ��
	//Mat& temp����ģ��ͼ��
	//Rect MatchRect������ƥ������
	//float StartAngle������ʼ�Ƕ�
	//float EndAngle���������Ƕ�
	//int cn��������
	//vector<TemplateMatchResult>& MResult����ƥ����
	//float Similarity�������ƶ���ֵ
	void matchTemplate_ExtendAnyAngle(Mat& img, Mat& temp, Rect MatchRect, float StartAngle, float EndAngle, float AngleStep, int cn, float Similarity, vector<TemplateMatchResult>& MResult);

	// ģ��ƥ�����Ƿ���չĿ��ͼ��
// ��ƥ���������ƥ��ʱ����ģ����תʱ�Ƿ񳬳�Ŀ��ͼ�������������Ҫ��ͼ����չ��������Ҫ
// �����Ҫ��չ����TRUE���������������߿��Ⱥ͸߶ȵ���չ��EW��EH�����򷵻�false��
//Mat temp����ģ��ͼ��
//Rect MatchRect����Ŀ��ͼ����Ҫ����ƥ�������
//int &EW����Ŀ��ͼ��Ҫ��չ�Ŀ��ȣ�������ӿ��ȣ�
//int &EH����Ŀ��ͼ��Ҫ��չ�ĸ߶ȣ��ϲ����Ӹ߶ȣ�
//int &EW2����Ŀ��ͼ��Ҫ��չ�Ŀ��ȣ��Ҳ����ӿ��ȣ�
//int& EH2����Ŀ��ͼ��Ҫ��չ�ĸ߶ȣ��ײ����ӿ��ȣ�
//BOOL MathchTemplate_IsExtendImage(Mat temp, Rect MatchRect, int &EW, int &EH);
	bool MathchTemplate_IsExtendImage(Mat Src, Mat temp, Rect MatchRect, int& EW, int& EH, int& EW2, int& EH2);

	// ģ��ƥ���ж�Ŀ��ͼ�����˫����չ�������ҷֱ���չEW�����·ֱ���չEH
// Mat Src����Ŀ��ͼ��
// Mat& Dst������չ���ͼ��
// int EW����Src�����չ�Ŀ���
// int EH����Src������չ�ĸ߶�
// int EW2����Src�Ҳ���չ�Ŀ���
// int EH2����Src�ײ���չ�ĸ߶�
// int Value�����������������ֵ
	void MathchTemplate_ExtendImage(Mat Src, Mat& Dst, int EW, int EH, int EW2, int EH2, int Value);

	//ͼ����ת
	void Rotate(const Mat& SrcImage, Point Pt[4], Mat& DestImage, double angle, int BValue);

	//Rect MatchRect����Ҫ����ƥ�������Χ
	void matchTemplate_mask(Mat& img, Mat& templ, Mat& result, Rect MatchRect, int cn, Mat& mask, bool IsRect);

	//ֻ���ƥ�������Ҫ��Ȼ��������Ŀ��
	void GetminMaxLoc(const Mat& src, int StartX, int EndX, int StartY, int EndY, double& minVal, double& maxVal, Point& MinPt, Point& MaxPt);

	//�����һ������
	void GetNextMaxLoc(Mat& result, int StartX, int EndX, int StartY, int EndY, Point maxLoc, int templatW, int templatH, double& NewMaxValue, Point& NewMaxLoc);

	//TMResult�п��ܴ������ڽǶ��µĶ����ֵ��
	void GetMaxMatchResult(vector<TemplateMatchResult> TMResult, TemplateMatchResult& MaxTMR);

	//TMResult�п��ܴ������ڽǶ��µĶ����ֵ�㣬�Ƚ����ֵ�ռ䷶Χ�ڵļ�ֵ�����������Ȼ���ٲ�����һ������
	void GetNextMaxMatchResult(vector<TemplateMatchResult>& TMResult, TemplateMatchResult& MaxTMR, int templatW, int templatH);

	//����ͼ�����صĺ��Լ�ƽ����
	void ComputeSumSq(Mat& temp, double& sum, double& sqsum);

	void DetectMaskContour_X(Mat& temp, Mat& mask, vector<MaskContour>& MaskContPts, int& Num, double& sum, double& sqsum);

	void DetectMaskContour_Y(Mat& temp, Mat& mask, vector<MaskContour>& MaskContPts, int& Num, double& sum, double& sqsum);
};

