#pragma once

#include <stdio.h>

/* Simple writer for VTK format files, used by Paraview and more. Writes an array to disk */

#define MAX_CHAR 80

template <class T>
class SimpleVTKWriter
{
public:
	SimpleVTKWriter(const char directory[], const T* data, const int nx, const int ny, const char varName[]) :  m_data(data), m_nx(nx), m_ny(ny)
	{
		int n = sprintf(m_directory, "%s", directory);

		if (n < 0) // some error checking 
		{
			printf("Error: directory too large for directory buffer in SimpleVTKWriter object.\n");
			exit(EXIT_FAILURE);
		}

		n = sprintf(m_variableName, "%s", varName);

		if (n < 0) // some error checking 
		{
			printf("Error: directory too large for directory buffer in SimpleVTKWriter object.\n");
			exit(EXIT_FAILURE);
		}
	}

	~SimpleVTKWriter();

	void write(const char* fileName) // write chosen array in constructor to disk, with chosen filename 
	{
		FILE* filePtr;

		char buffer[MAX_CHAR];

		sprintf(buffer, "%s\\%s.vti", m_directory, fileName); // add vti extension for paraview to recognise it, forgoing error checking this time ... and directory location

		filePtr = fopen(buffer, "w");

		writeHeader(filePtr);
		writeArray(filePtr);
		writeFooter(filePtr);

		fclose(filePtr);
	}

private:

	void writeArray(FILE *outputFile) // writes the bulk of the xml file - the data in question
	{
		fprintf(outputFile, "     <DataArray type=\"Float32\" Name=\"%s\" format=\"ascii\">\n", m_variableName);

		for (int x = 0; x < m_nx; x++)
		{
			for (int y = 0; y < m_ny; y++)
					fprintf(outputFile, "%G\t", m_data[x * m_ny + y]);
			fprintf(outputFile, "\n");
		}

		fprintf(outputFile, "      </DataArray>\n");
	}

	void writeFooter(FILE *outputFile)
	{
		//closing xml tags at the footer
		fprintf(outputFile, "    </PointData>\n");
		fprintf(outputFile, "   <CellData>\n");
		fprintf(outputFile, "   </CellData>\n");
		fprintf(outputFile, "  </Piece>\n");
		fprintf(outputFile, " </ImageData>\n");
		fprintf(outputFile, "</VTKFile>\n");
	}

	void writeHeader(FILE *outputFile)
	{
		//header tags, e.g. domain size, data types
		fprintf(outputFile, "<?xml version=\"1.0\"?>\n");
		fprintf(outputFile, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
		fprintf(outputFile, " <ImageData WholeExtent=\"0 %i 0 %i 0 %i\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n", m_nx - 1, m_ny - 1, 0);
		fprintf(outputFile, "  <Piece Extent=\"0 %i 0 %i 0 %i\">\n", m_nx - 1, m_ny - 1, 0);
		fprintf(outputFile, "    <PointData> Scalars=\"%s\" Vectors=\"\">\n",m_variableName);

	}

	char m_directory[MAX_CHAR];
	char m_variableName[MAX_CHAR];
	const int m_nx;
	const int m_ny;

	const T* m_data;

};


