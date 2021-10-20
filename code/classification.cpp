#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/classification.hpp>
#include <vitis/ai/demo.hpp>

int main(int argc, char* argv[]) {
    std::string image_name = argv[1];
    auto image = cv::imread(image_name);
    auto network = vitis::ai::Classification::create("customcnn");
    auto result = network->run(image);
    cout << "Classification result:" << endl; 
    for (const auto &r : result.scores){
	cout << result.lookup(r.index) << ": " << r.score << endl;
    }
}
