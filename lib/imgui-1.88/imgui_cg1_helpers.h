/**
 * © 2022, CGVR (https://cgvr.informatik.uni-bremen.de/),
 * Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)
 *
 * This file contains some ImGui specific helper functions
 * for CG1
 */

#pragma once

// Include the GUI:
#include "imgui.h"


namespace ImGuiCG1
{
	/**
	 * Creates a table to show the given matrix as a table in ImGUI.
	 */

    void DragMat4f(const char* name, float* data, const bool editable = false, const ImVec4 bgColor = ImVec4(0.2f, 0.2f, 0.2f, 1.f), const float dragSpeed = 0.005f, const float dragMin = -1.f, const float dragMax = 1.f){
		if(ImGui::BeginTable(name, 4, editable ? 0 : ImGuiTableFlags_Borders )){
			for (int y = 0; y < 4; y++){
				for (int x = 0; x < 4; x++){
					ImGui::TableNextColumn();
					ImGui::PushID(x+y*4);
					ImGui::PushItemWidth(-FLT_MIN);
					if(editable)
                        ImGui::DragFloat("#", &data[y+x*4], dragSpeed, dragMin, dragMax);
                    else {
                        ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(bgColor));
                        ImGui::Text("%.3lf", data[y+x*4]);
                    }
					ImGui::PopID();
				}
			}
			ImGui::EndTable();
		}
	}

	/**
	 * Creates a table to show the given Vec4f as a single-row table in ImGUI.
	 */

    void DragVec4f(const char* name, float* data,  const bool editable = false, const ImVec4 bgColor = ImVec4(0.2f, 0.2f, 0.2f, 1.f), const float dragSpeed = 0.005f, const float dragMin = -1.f, const float dragMax = 1.f){
		if(ImGui::BeginTable(name, 4, editable ? 0 : ImGuiTableFlags_Borders)){
			for (int i = 0; i < 4; i++){
				ImGui::TableNextColumn();
				ImGui::PushID(i);
				ImGui::PushItemWidth(-FLT_MIN);
				if(editable)
                    ImGui::DragFloat("#", (data + i), dragSpeed, dragMin, dragMax);
				else {
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(bgColor));
                    ImGui::Text("%.3lf", *(data + i));
				}
				ImGui::PopID();
			}
			ImGui::EndTable();
		}
	}

	/**
	 * Just shows a float value and prints a label for it in ImGUI.
	 */

    void TextFloat(const char* name, float value, ImVec4 color = ImVec4(0.6f,0.6f,0.6f,1.f)){
        ImGui::TextColored(color,"%s", name);
		ImGui::SameLine();
		ImGui::Text("%.3lf", value);
	}

	/**
	 * Just shows a bool value and prints a label for it in ImGUI.
	 */

    void TextBool(const char* name, bool value, ImVec4 color = ImVec4(0.6f,0.6f,0.6f,1.f)){
        ImGui::TextColored(color,"%s", name);
		ImGui::SameLine();
		ImGui::Text("%s", value ? "True" : "False");
	}
}
