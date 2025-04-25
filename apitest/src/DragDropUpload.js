// src/components/DragDropUpload.js
import React from 'react';
import { useDropzone } from 'react-dropzone';
import PropTypes from 'prop-types';

const DragDropUpload = ({ onUpload }) => {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/json': ['.json'],
      'application/yaml': ['.yaml', '.yml'],
      'text/yaml': ['.yaml', '.yml']
    },
    maxSize: 5 * 1024 * 1024, // 5MB
    multiple: false,
    onDrop: async files => {
      try {
        if (files[0]) await onUpload(files[0]);
      } catch (error) {
        console.error('Drop error:', error);
      }
    }
  });

  return (
    <div 
      {...getRootProps()}
      style={{
        border: '2px dashed #4CAF50',
        borderRadius: '4px',
        padding: '20px',
        textAlign: 'center',
        backgroundColor: isDragActive ? '#e8f5e9' : '#f5f5f5',
        cursor: 'pointer',
        marginRight: '20px',
        minWidth: '300px'
      }}
    >
      <input {...getInputProps()} />
      <p style={{ margin: 0, color: '#333' }}>
        {isDragActive 
          ? 'Drop the OpenAPI spec here' 
          : 'Drag & drop OpenAPI spec (JSON/YAML), or click to select'}
      </p>
    </div>
  );
};
DragDropUpload.propTypes = {
  onUpload: PropTypes.func.isRequired
};
export default DragDropUpload;